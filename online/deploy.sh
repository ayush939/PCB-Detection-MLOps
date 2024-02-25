az ml online-endpoint create  -f ./online-endpoint.yaml
az ml online-deployment create --name blue --endpoint  pcb-detection-online -f ./blue-deployment.yaml --all-traffic
az ml online-endpoint invoke --name PCB-detection-online --request-file ../../Inference/pdfs/-b_AN00985600_Panel-Drawing_4er.pdf
