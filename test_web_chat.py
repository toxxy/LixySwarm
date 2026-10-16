"""The web bridge must stay stateless and reject empty/error model responses."""
import unittest
from unittest.mock import patch

from api import chat_handler
try:
    import api.main as api
except ModuleNotFoundError:
    api=None


class WebHandlerTests(unittest.TestCase):
    def test_only_current_question_is_forwarded_without_personal_context(self):
        result={'answer':'París','provider_id':'a'*64,'model_id':'b'*64,'receipt':{'signature':'checked'}}
        with patch.object(chat_handler.NodeController,'ask_chat',return_value=result) as call:
            value=chat_handler.chat('Pregunta pública')
        self.assertEqual(value['response'],'París')
        args=call.call_args.args
        self.assertEqual(args[0],[{'role':'user','content':'Pregunta pública'}])
        self.assertIs(args[1],False);self.assertEqual(len(args[2]),32)

    def test_unavailable_and_busy_do_not_return_success(self):
        with patch.object(chat_handler.NodeController,'ask_chat',side_effect=RuntimeError('private path canary')):
            self.assertEqual(chat_handler.chat('Pregunta')['status'],'unavailable')
        chat_handler._slot.acquire()
        try:self.assertEqual(chat_handler.chat('Pregunta')['status'],'busy')
        finally:chat_handler._slot.release()

    def test_byte_limits_are_checked_before_any_request(self):
        with patch.object(chat_handler.NodeController,'ask_chat',side_effect=AssertionError('must not run')):
            for text in [' ','é'*2049,'x'*4097]:self.assertEqual(chat_handler.chat(text)['status'],'invalid')


@unittest.skipIf(api is None,'FastAPI is validated in the deployed API environment')
class WebEndpointTests(unittest.TestCase):
    def test_history_does_not_share_a_visitors_question(self):
        with patch.object(chat_handler,'chat',return_value={'status':'ok','response':'private-answer-canary'}):
            api.chat(api.ChatRequest(message='private-question-canary'))
        self.assertEqual(api.chat_history()['data'],[])

    def test_null_and_error_outputs_are_not_success(self):
        for result in [{'status':'error','response':None},{'status':'ok','response':None},{'status':'ok','response':''}]:
            with patch.object(chat_handler,'chat',return_value=result):
                self.assertEqual(api.chat(api.ChatRequest(message='test')).status_code,503)

    def test_busy_returns_retryable_http_error(self):
        with patch.object(chat_handler,'chat',return_value={'status':'busy'}):
            self.assertEqual(api.chat(api.ChatRequest(message='test')).status_code,429)


if __name__=='__main__':unittest.main()
