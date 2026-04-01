import requests
import yaml
import json
import os

class MetadataManager:
    """
    元数据管理类
    用于与Dify API交互，创建和绑定元数据字段
    """
    def __init__(self):
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        self.dify_config = config["dify"]
        self.meta_config = config.get("meta", {})
        self.base_url = f"https://api.dify.ai/v1/datasets/{self.dify_config['dataset_id']}"
        self.headers = {
            "Authorization": f"Bearer {self.dify_config['api_key']}",
            "Content-Type": "application/json"
        }
    
    def create_metadata_field(self, field_type: str, field_name: str) -> str:
        """
        创建元数据字段，返回metadata_id

        :param field_type: 元数据字段类型，如"string", "number", "date"等
        :param field_name: 元数据字段名称，用于标识字段
        :return: 创建成功后的metadata_id
        """
        url = f"{self.base_url}/metadata"
        payload = {
            "type": field_type,
            "name": field_name
        }
        try:
            print(f"创建元数据字段: {field_name}, 请求体: {payload}")
            response = requests.post(url, headers=self.headers, data=json.dumps(payload))
            response.raise_for_status()
            result = response.json()
            print(f"创建成功，返回ID: {result.get('id')}")
            return result
        except requests.exceptions.HTTPError as e:
            error_details = response.json() if response.content else str(e)
            print(f"创建失败: HTTP {response.status_code}, 详情: {error_details}")
            if response.status_code == 400:
                if "already exists" in str(error_details).lower():
                    print(f"提示: 元数据字段'{field_name}'已存在，请在config.yaml中添加其ID")
                elif "required parameter" in str(error_details).lower():
                    print(f"提示: 请求缺少必填参数，请检查'type'和'name'是否正确传递")
            raise Exception(f"创建元数据字段失败: {response.status_code} - {error_details}")
    
    def bind_metadata_to_document(self, document_id: str, metadata_list: list) -> dict:
        """
        绑定元数据到文档（自动创建缺失的元数据字段）

        :param document_id: 文档ID，用于指定要绑定元数据的文档
        :param metadata_list: 元数据列表，每个元素为包含'name'和'value'的字典
        :return: 绑定成功后的响应内容
        """
        metadata_fields = self.meta_config.get("fields", [])
        if not metadata_fields:
            print("未配置元数据字段，跳过绑定")
            return

        # 创建新的元数据列表，避免修改原始输入
        formatted_metadata = []
        for field in metadata_fields:
            metadata_id = field.get("id")
            field_name = field["name"]

            if not metadata_id:
                print(f"错误: 元数据字段'{field_name}'未配置ID，请在config.yaml中添加")
                continue

            # 从传入的metadata_list中查找对应的值
            value = next((item['value'] for item in metadata_list if item['name'] == field_name), "")
            formatted_metadata.append({
                "id": metadata_id,
                "value": value,
                "name": field_name
            })

        if not formatted_metadata:
            print("警告: 未找到有效的元数据配置，无法执行绑定")
            return {}

        # 构建完整的URL
        # In bind_metadata_to_document method
        # Old URL (404 error)
        # url = f"{self.base_url}/documents/{document_id}/metadata"
        
        # 修正URL：使用base_url(已包含dataset_id) + 批量元数据端点
        url = f"{self.base_url}/documents/metadata"
        payload = {
            "operation_data": [{
                "document_id": document_id,
                "metadata_list": formatted_metadata
            }]
        }
        # 添加调试信息
        print(f"发送元数据绑定请求: URL={url}, Payload={payload}")
        response = requests.post(url, headers=self.headers, data=json.dumps(payload))
        # 打印响应状态
        print(f"绑定请求响应: 状态码={response.status_code}, 响应内容={response.text}")
        if response.status_code != 200:
            raise Exception(f"绑定元数据失败: {response.status_code} - {response.text}")
        return response.json()
    
    def bind_metadata(self, document_id, metadata_list):
        self.bind_metadata_to_document(document_id, metadata_list)

if __name__ == "__main__":
    manager = MetadataManager()
    meta_fields = manager.meta_config.get("fields", [])
    created_fields = []
    for field in meta_fields:
        try:
            created_metadata = manager.create_metadata_field("string", field["name"])
            created_fields.append(created_metadata)
            print(f"创建元数据字段成功: {created_metadata['name']} (ID: {created_metadata['id']})")
        except Exception as e:
            print(f"创建元数据字段 {field['name']} 失败: {str(e)}")

    # 2. 提前定义元数据列表
    filename = 'example.png'
    metadata = [
        {'name': 'author', 'value': 'default_author'},
        {'name': 'doc_source', 'value': "example.png"}
    ]

    test_document_id = "9a0634ea-8575-45f5-b3be-c4f576ca9251"
    try:
        print(f"调试信息: 尝试绑定元数据到文档ID: {test_document_id}")
        print(f"调试信息: API端点: {manager.base_url}/documents/{test_document_id}/metadata")
        result = manager.bind_metadata(test_document_id, metadata)
        print(f"元数据绑定成功: {result}")
    except Exception as e:
        print(f"元数据绑定失败: {str(e)}")