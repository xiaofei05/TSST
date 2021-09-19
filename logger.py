import logging

class Log(object):
    def __init__(self, name=None, file_path=''):

        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        fh = logging.FileHandler(file_path, 'a+', encoding="utf-8")
        fh.setLevel(logging.INFO)
 
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
 
        formatter = logging.Formatter('%(asctime)s - %(message)s')

        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
 
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
 
        fh.close()
        ch.close()


    def getlog(self):
        return self.logger
