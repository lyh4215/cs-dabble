import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from exp3_association import parse_trace,mine_rules,candidates,rates

class TestRules(unittest.TestCase):
    def test_pid_variants(self):
        with TemporaryDirectory() as d:
            p=Path(d)/'trace'
            p.write_text('1234 openat(1) = 2\n[pid 99] getrusage(0) = 0\n uname(0) = 0\n')
            self.assertEqual(parse_trace(p), {'openat','getrusage','uname'})
    def test_association(self):
        corpus=[{'read','write','close'},{'read','write'},{'read','write','getrusage'},{'read','write','close'}]
        rules,_=mine_rules(corpus,.5,.8)
        c=candidates({'read'},rules)
        self.assertIn('write',c)
        self.assertNotIn('getrusage',c)
        report=rates({'read'},{'write'},{'getrusage'},rules)
        self.assertEqual(report['legit_recall'],1)
        self.assertEqual(report['noise_accept_rate'],0)

if __name__=='__main__':unittest.main()
