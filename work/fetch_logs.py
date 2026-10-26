from pathlib import Path
import requests, sys
from kaggle import api
from kaggle.api.kaggle_api_extended import ApiListKernelSessionOutputRequest
slug=sys.argv[1] if len(sys.argv)>1 else 'namas'
out=Path(__file__).resolve().parent/('gpu-logs' if slug=='namas' else ('full-results' if slug=='namas-full-quality' else 'training-results'))
out.mkdir(exist_ok=True)
with api.build_kaggle_client() as client:
    token=None
    while True:
        req=ApiListKernelSessionOutputRequest()
        req.user_name=__import__('os').environ['KAGGLE_USERNAME']
        req.kernel_slug=slug
        req.page_size=200
        if token: req.page_token=token
        result=client.kernels.kernels_api_client.list_kernel_session_output(req)
        if result.log: (out/'notebook.log').write_text(result.log,encoding='utf-8')
        for item in result.files:
            if item.file_name.endswith(('.log','.json')) or ('comparison_' in item.file_name and item.file_name.endswith('.jpg')):
                r=requests.get(item.url, timeout=60)
                r.raise_for_status()
                (out/Path(item.file_name).name).write_bytes(r.content)
                print(item.file_name)
        token=result.next_page_token
        if not token: break
