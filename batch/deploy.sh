az ml batch-endpoint create --name PCB-detection --file endpoint.yaml
az ml batch-deployment create --name batch-dp --file batch-deployment-yaml --set-default --endpoint-name PCB-detection
az ml batch-endpoint invoke --name PCB-detection --input ../../testPanels/images/test/ --output-path azureml://datastores/workspaceblobstore/paths/tests/output