import requests
import json
import yaml
import os

class DifyUploader:
    def __init__(self):
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)["dify"]
        
        self.base_url = f"https://api.dify.ai/v1/datasets/{self.config['dataset_id']}"
    
    def list_datasets(self, page=1, limit=20):
        '''
        查询Dify知识库列表
        :param page: 页码
        :param limit: 每页数量
        :return: 知识库列表
        '''

        headers = {
            "Authorization": f"Bearer {self.config['api_key']}",
            "Content-Type": "application/json"
        }

        params = {
            "page": page,
            "limit": limit
        }

        response = requests.get(
            "https://api.dify.ai/v1/datasets",
            headers=headers,
            params=params
        )
        return response.json()
    
    def upload_document_by_text(self, segments):
        '''
        上传文档至Dify
        :param segments: 预处理后的文本段落列表
        :return: Dify分配的文档ID
        '''
        headers = {
            "Authorization": f"Bearer {self.config['api_key']}",
            "Content-Type": "application/json"
        }

        response = None
        for i, content in enumerate(segments):
            payload = {
                "name": f"segment_{i}.txt",
                "text": content,
                "indexing_technique": self.config.get("indexing_technique", "default"),
                "process_rule": {"mode": "automatic"},
                "doc_form": "text_model",
                "doc_language": "Chinese"
            }
            data = {
                "content": content
            }
            response = requests.post(
                f"{self.base_url}/document/create-by-text",
                headers=headers,
                data=json.dumps(payload)
            )
            if response.status_code != 200:
                raise Exception(f"上传失败: {response.text}")
            print(f"Segment {i} uploaded: {response.status_code}")
            if response is None:
                raise RuntimeError("No segments were processed")
        if response is None:
            raise RuntimeError("No segments were processed")
        return response.json()["document"]["id"]
    
if __name__ == "__main__":
    uploader = DifyUploader()
    # 查询所有数据集
    datasets = uploader.list_datasets()
    print("可用数据集列表:")
    for ds in datasets.get('data', []):
        print(f"ID: {ds['id']}, 名称: {ds['name']}")
    mock_text = ['WMW. 登机牌 BOARDING PASS 登机牌 BOARDING PASS 日期 DATE 舱位 CLASS 序号 SERIAL NO. 座位号 SEAT NO. 日期DATE 舱位 CLASS 序号 SERIALNO. 座位号 SEAT NO. 航班 FLIGHT 航班FLIGHT MU2379 03DEC W 035 MU 2379 03DEC W 035 始发地 FROM 登机口 GATE 登机时间 BDT 始发地 FROM 登机口 GATE 登机时间 BDT 目的地 TO 目的地TO 福州 G11 福州 TAIYUAN TAIYUAN G11 FUZHOU 身份识别IDNO. FUZHOU 姓名NAME 姓名NAME 身份识别IDNO. ZHANGQIWEI ZHANGQIWEI 票号TKT NO 票号TKTNO. 张祺伟 张祺伟 票价FARE ETKT7813699238489/1 票价FARE ETKT7813699238489/1 登机口于起飞前1O分钟关闭GATESCLOSE1O MINUTES BEFORE DEPARTURE TIME 登机口于起飞前1O分钟关闭GATESCLOSE1OMINUTESBEFOREDEPARTURETIME']

    uploader = DifyUploader()
    # 查询知识库列表并仅显示id和name字段
    datasets = uploader.list_datasets()
    # 提取id和name字段
    filtered_datasets = [{'id': item['id'], 'name': item['name']} for item in datasets.get('data', [])]
    print("知识库列表:", filtered_datasets)

    document_id = uploader.upload_document_by_text(mock_text)
    print(f"Document ID: {document_id}")