# Sockeye Thrift RPC service

## Installing prerequisites
The Sockeye Thrift RPC server requires a few prerequisites be installed in addition to the usual sockeye prerequisites.
You can install them by running the following command from the source root directory:
```bash
pip install -r contrib/rpc/requirements.txt
``` 


## Code generation instructions
Our RPC server requires to generate a few python files.  Again from the source root directory run:
```bash
thrift -gen py -out contrib/rpc   contrib/rpc/thrift/translate_service.thrift
```
You should now have a folder in your contrib/rpc directory named 'translate_service'.  This folder contains all the
generated code required to start a rpc thrift translation service.

## Setting your PYTHONPATH
Before starting the server you may need to set your pythonpath to properly load sockeye contrib packages.
From the source root dir run:
```bash
export PYTHONPATH=`pwd`:$PYTHONPATH
```

## Starting the server
You can now start service requests.  Just start start rpc_server.py with the same command line arguments you would use
to run translate.py.  For example:
```bash
python contrib/rpc/rpc_server.py --use-cpu  -m model_folder
```

## Testing the server
You can now test the server by running:
```bash
python contrib/rpc/rpc_client.py
```