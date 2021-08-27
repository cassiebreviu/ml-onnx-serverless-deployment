import logging
import onnxruntime as rt
import numpy as np
import azure.functions as func
from skl2onnx.common.data_types import StringTensorType


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    x_data = req.params.get('name')
    if not x_data:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            x_data = req_body.get('name')

    if x_data:
        # Compute the prediction with ONNX Runtime
        
        logging.info(type(x_data.strip()))
        sess = rt.InferenceSession("pipeline_quality.onnx")
        input_name = sess.get_inputs()[0].name
        logging.info(input_name)
        label_name = sess.get_outputs()[0].name
        logging.info(label_name)
        # arg0: List[str], arg1: Dict[str, object], arg2: onnxruntime.capi.onnxruntime_pybind11_state.RunOptions
        pred_onx = sess.run([label_name], {input_name: [x_data]})[0]
        print(f"{pred_onx[0]}")
        return func.HttpResponse(f"{pred_onx}")
    else:
        return func.HttpResponse(
             "This HTTP triggered function executed successfully. Pass a name in the query string or in the request body for a personalized response.",
             status_code=200
        )
