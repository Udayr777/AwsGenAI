import boto3
import json
import base64
import os

prompt_data = """ 
A man who lose hope in his life.            
"""
prompt_templates = [{"text": prompt_data,"weight":1}]
bedrock = boto3.client(service_name='bedrock-runtime')

payload = {
    "text_prompts": prompt_templates,
    "cfg_scale": 10,
    "seed": 0,
    "steps": 50,
    "width": 512,
    "height": 512
}

body = json.dumps(payload)
model_id = "stability.stable-diffusion-xl-v1"
response = bedrock.invoke_model(
    modelId=model_id,
    body=body,
    contentType="application/json",
    accept="application/json"
)


response_body = json.loads(response.get("body").read())
print(response_body)
artifact = response_body.get("artifacts")[0]
image_encoded = artifact.get("base64").encode("utf-8")
image_bytes = base64.b64encode(image_encoded)
 
#e save image to the file in the output directory
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
file_name = f"{output_dir}/stableRocks.png"
with open(file_name, "wb") as f:
    f.write(image_bytes)
