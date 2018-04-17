# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not
# use this file except in compliance with the License. A copy of the License
# is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

"""
Test client for the sockeye RPC server.
"""

from contrib.rpc.translate_service.Translate import Client

from tornado.ioloop import IOLoop
from tornado import gen
from thrift import Thrift
from torthrift.pool import TStreamPool
from thrift.protocol.TBinaryProtocol import TBinaryProtocolFactory
from torthrift.client import PoolClient

ioloop = IOLoop.instance()


@gen.coroutine
def test():
    try:
        transport = TStreamPool('127.0.0.1', 9095, max_stream=10)
        client = PoolClient(Client, transport, TBinaryProtocolFactory())
        for i in range(0, 20):
            res = yield client.translate('Die USA und Großbritannien berichten von einer mutmaßlichen weltweiten Cyberattacke.')
            res = res.replace('@@ ', '')
            print(res)
            res = yield client.translate('Von der Regierung in Moskau unterstützte Hacker-Gruppen hätten Router, Switches und Firewalls infiziert, so Behörden beider Länder.')
            res = res.replace('@@ ', '')
            print(res)
    except Thrift.TException as ex:
        print("%s" % ex.message)
    ioloop.stop()


def main():
    ioloop.add_callback(test)
    ioloop.start()


if __name__ == "__main__":
    main()
