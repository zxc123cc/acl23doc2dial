import os
import random
import hashlib
import requests
import json
from modelscope.msdatasets import MsDataset
from modelscope.utils.constant import DownloadMode
from transformers import MT5Tokenizer
import time

generation_tokenizer = MT5Tokenizer.from_pretrained(os.path.join('pretrain_storage/gen_pretrain_model', 'generation'))
convert_dict = {}

def handle_query(query,fromLang='zh',toLang='en'):
    tmp = query.replace('<user>','<agent>')
    tmp = tmp.replace('<last_turn>','')
    text_list = tmp.split('<agent>')
    result = ''
    if len(text_list)>=1:
        result = '<last_turn> ' + baidu_translate(text_list[0],fromLang,toLang)
    if len(text_list)>=2:
        result = result + ' <agent> ' + baidu_translate(text_list[1],fromLang,toLang)
    if len(text_list)>=3:
        result = result + ' <user> ' + baidu_translate(text_list[2],fromLang,toLang)

    return result


def handle_passages(passage,fromLang='zh',toLang='en'):
    if fromLang != 'zh':
        passage = generation_tokenizer.decode(
            generation_tokenizer([passage], add_special_tokens=False,return_tensors='pt')['input_ids'][0][:500]
        )
    else:
        passage = passage[:min(500,len(passage))]
    result = baidu_translate(passage,fromLang,toLang)
    return json.dumps([result]).replace('/ /','//')


def handle_response(response,fromLang='zh',toLang='en'):
    result = baidu_translate(response.replace('<response>',''),fromLang,toLang)
    result = '<response> ' + result
    return result


def translate_dataset(fromLang,toLang,save_path=None):
    f = open(save_path, 'w', encoding='utf-8')
    if fromLang == 'zh':
        dataset = MsDataset.load(
            'DAMO_ConvAI/ZhDoc2BotDialogue',
            download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)
    if fromLang == 'en':
        dataset = MsDataset.load(
            'DAMO_ConvAI/EnDoc2BotDialogue',
            download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)
    num = 0
    print('total len: ',len(dataset))
    for data in dataset:
        num += 1
        print(num)
        if num <12000:
            continue
        translated_query = handle_query(data['query'],fromLang,toLang)
        print(data['query'])
        print(translated_query)
        if fromLang=='en':
            passage = data['passages'][2:-2]
        else:
            passage = json.loads(data['passages'])[0]
        translated_passages = handle_passages(passage,fromLang,toLang)
        print(data['passages'])
        print(translated_passages)
        translated_response = handle_response(data['response'],fromLang,toLang)
        print(data['response'])
        print(translated_response)

        f.write(json.dumps({
            'query': translated_query,
            'passages': translated_passages,
            'response': translated_response
        }, ensure_ascii=False) + '\n')


def baidu_translate(en_str,fromLang='zh',toLang='en'):
    en_str = en_str.strip()
    if en_str in convert_dict:
        print('use cache...')
        return convert_dict[en_str]
    time.sleep(1)
    api_url = 'http://api.fanyi.baidu.com/api/trans/vip/translate'
    # 百度翻译appid
    appid = ''
    # api的密码
    secretKey = ''
    salt = random.randint(32768, 65536)
    # sign = get_md5(appid + en_str + str(salt) + secretKey)
    sign = appid + en_str + str(salt) + secretKey
    sign = hashlib.md5(sign.encode()).hexdigest()
    # 可以设置翻译的语言
    api_data = {
        'q': en_str,
        'from': fromLang,
        'to': toLang,
        'appid': appid,
        'salt': salt,
        'sign': sign
    }
    req_get = requests.get(api_url, api_data)
    # 结果的位置可能不同
    print(req_get.json())
    # if 'trans_result' not in req_get.json():
    #     return ''
    result = req_get.json()['trans_result'][0]['dst']
    convert_dict[en_str] = result
    return result

def tmp():
    file_path = 'DAMO_ConvAI/translate_zh2vi.json'
    with open(file_path,'r',encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines[:10]:
            item = json.loads(line)
            print(json.loads(item['passages']))


if __name__ == '__main__':
    # 手动录入翻译内容，q存放
    # q = "你好帅"
    # result_fra = baidu_translate(q,fromLang='zh',toLang='fra')
    # result_vie = baidu_translate(q,fromLang='zh',toLang='vie')
    # print("原句:"+q)
    # print("翻译为法语：",result_fra)
    # print("翻译为越南语：",result_vie)
    translate_dataset(fromLang='en', toLang='vie', save_path='DAMO_ConvAI/translate_en2vi_12000.json')
