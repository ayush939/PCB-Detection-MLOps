import argparse
from pathlib import Path
import pandas as pd
import json
import os
import mlflow
from azure.identity import DefaultAzureCredential, ManagedIdentityCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
#from azureml.core.model import Model
from azure.ai.ml.constants import AssetTypes

parser = argparse.ArgumentParser(description='Model validation')

parser.add_argument('--experiment_name', type=str, default=None,
                    help='Name of the experiment to compare the current metric with previous runs')

parser.add_argument('--current_metrics', type=str, default=None,
                    help='Path to the current metric values')

parser.add_argument('--modelPath', type=str,
                    help='Path to the model')

# Optional positional argument
parser.add_argument('--modelName', type=str, default = "yolov5",
                    help='Name of the model to be registered as')
args = parser.parse_args()


runs = mlflow.search_runs(
    experiment_names=[ args.experiment_name ],
    output_format="list"
)

f = open(Path(args.current_metrics) / "data.json", 'r')
data = json.load(f)
Fscore = data["F-score"]
#print("Data", data)
flag=0
a=0
if len(runs)!= 0:
    for i in range(len(runs)):
        metrics = runs[i].data.metrics
        #print(metrics)
        try:
            if metrics["F-score"] < Fscore:
               
                flag=1
            else:
                job_name = runs[i].data.tags["azureml.pipeline"]
                a=metrics["F-score"] 
                flag=0
                break
        except:
            pass

    if  flag:
        print("Registering model ;)")
        
        #credential = DefaultAzureCredential()
        ml_client = None
        try:
            ml_client = MLClient.from_config(credential)
        except Exception as ex:
            print(ex)
            # Enter details of your AML workspace
            subscription_id="f1dec604-2ef2-4271-af97-21bb6a2eea00"
            resource_group="rg-cdtmlopsv2-0670dev"
            workspace="mlw-cdtmlopsv2-0670dev"
            #client_id = os.environ.get('DEFAULT_IDENTITY_CLIENT_ID')
            client_id = "5b725f2a-453a-439d-b1b8-6e4cfd924768"
            
            credential = ManagedIdentityCredential(client_id=client_id)
            #token = credential.get_token('https://storage.azure.com/')
            ml_client = MLClient(credential, subscription_id, resource_group, workspace)
        
        cloud_model = Model(
            path=args.modelPath + "/best.pt",
            name=args.modelName,
            type=AssetTypes.CUSTOM_MODEL,
            description="PCB detection model",
        )
        ml_client.models.create_or_update(cloud_model)
        print("Model Registered")

    else:
        print(f"F-score:{Fscore} is less than {a} from job {job_name}. Hence, NOT registering the model! HINT: You need to train the model again with good hyperparameters or a better dataset.", )

else:
    print(f"Registering the first model of the experiment {args.experiment_name} ;)")

    #credential = DefaultAzureCredential()
    ml_client = None
    try:
        ml_client = MLClient.from_config(credential)
    except Exception as ex:
        print(ex)
        # Enter details of your AML workspace
        subscription_id="f1dec604-2ef2-4271-af97-21bb6a2eea00"
        resource_group="rg-cdtmlopsv2-0670dev"
        workspace="mlw-cdtmlopsv2-0670dev"
        #client_id = os.environ.get('DEFAULT_IDENTITY_CLIENT_ID')
        client_id = "5b725f2a-453a-439d-b1b8-6e4cfd924768"
        
        credential = ManagedIdentityCredential(client_id=client_id)
        #token = credential.get_token('https://storage.azure.com/')
        ml_client = MLClient(credential, subscription_id, resource_group, workspace)

    cloud_model = Model(
        path=args.modelPath + "/best.pt",
        name=args.modelName,
        type=AssetTypes.CUSTOM_MODEL,
        description="PCB detection model",
    )
    ml_client.models.create_or_update(cloud_model)
    print("Model Registered")


"""
#df = pd.read_csv(Path(args.datapath) / 'metric.csv')
#pd.concat([df, pd.DataFrame([data])]).to_csv(Path(args.datapath) / 'metric.csv', index=False)
pd.DataFrame([data]).to_csv(Path(args.datapath) / 'metric.csv', index=False)
"""