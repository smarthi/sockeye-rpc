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
Basic RPC server utilizing a single process and thread to serve translations from sockeye.
"""

import argparse
import threading
from contextlib import ExitStack

import mxnet as mx
from thrift.protocol.TBinaryProtocol import TBinaryProtocolFactory
from tornado import gen
from tornado.ioloop import IOLoop
from torthrift.server import TTornadoServer
from torthrift.transport import TIOStreamTransportFactory

from contrib.rpc.translate_service.Translate import Processor
from sockeye import inference, arguments
from sockeye.log import setup_main_logger
from utils import check_condition, get_num_gpus, acquire_gpus

logger = setup_main_logger(__name__, file_logging=False)


class Handler(object):

    def __init__(self, translator):
        self.translator = translator
        self.request_counter = 0

    @gen.coroutine
    def translate(self, source):
        inputs = [inference.make_input_from_plain_string(self.request_counter, source)]
        translations = self.translator.translate(inputs)
        print('translate called')
        print('Thread: ' + str(threading.current_thread()))
        self.request_counter = self.request_counter + 1
        return translations[0].translation

    @gen.coroutine
    def batch_translate(self, source):
        inputs = []
        for s in source:
            translation_input = inference.make_input_from_plain_string(self.request_counter, s)
            self.request_counter = self.request_counter + 1
            inputs.append(translation_input)

        translations = self.translator.translate(inputs)
        print('translate called')
        print('Thread: ' + str(threading.current_thread()))
        self.request_counter = self.request_counter + 1
        return [t.translation for t in translations]


class SockeyeRpcServer(object):

    def __init__(self, translator):
        self.translator = translator

    def serve(self):
        handler = Handler(self.translator)
        processor = Processor(handler)
        tfactory = TIOStreamTransportFactory()
        protocol = TBinaryProtocolFactory()

        server = TTornadoServer(processor, tfactory, protocol)
        server.bind(20000)
        server.start(1)  # MXNet requires we use a single thread, single process.
        IOLoop.instance().start()


def main():
    params = argparse.ArgumentParser(description='Translate CLI')
    arguments.add_translate_cli_args(params)
    args = params.parse_args()

    with ExitStack() as exit_stack:
        if args.use_cpu:
            context = mx.cpu()
        else:
            num_gpus = get_num_gpus()
            check_condition(num_gpus >= 1,
                            "No GPUs found, consider running on the CPU with --use-cpu "
                            "(note: check depends on nvidia-smi and this could also mean that the nvidia-smi "
                            "binary isn't on the path).")
            check_condition(len(args.device_ids) == 1, "cannot run on multiple devices for now")
            gpu_id = args.device_ids[0]
            if args.disable_device_locking:
                if gpu_id < 0:
                    # without locking and a negative device id we just take the first device
                    gpu_id = 0
            else:
                gpu_ids = exit_stack.enter_context(acquire_gpus([gpu_id], lock_dir=args.lock_dir))
                gpu_id = gpu_ids[0]

            context = mx.gpu(gpu_id)

        models, source_vocabs, target_vocab = inference.load_models(
            context=context,
            max_input_len=args.max_input_len,
            beam_size=args.beam_size,
            batch_size=args.batch_size,
            model_folders=args.models,
            checkpoints=args.checkpoints,
            softmax_temperature=args.softmax_temperature,
            max_output_length_num_stds=args.max_output_length_num_stds,
            decoder_return_logit_inputs=args.restrict_lexicon is not None,
            cache_output_layer_w_b=args.restrict_lexicon is not None)

        translator = inference.Translator(context=mx.cpu(),
                                          ensemble_mode=args.ensemble_mode,
                                          bucket_source_width=args.bucket_width,
                                          length_penalty=inference.LengthPenalty(args.length_penalty_alpha,
                                                                                 args.length_penalty_beta),
                                          models=models,
                                          source_vocabs=source_vocabs,
                                          target_vocab=target_vocab,
                                          restrict_lexicon=None,
                                          store_beam=False,
                                          strip_unknown_words=args.strip_unknown_words)

        rpc_server = SockeyeRpcServer(translator)
        rpc_server.serve()


if __name__ == '__main__':
    main()
