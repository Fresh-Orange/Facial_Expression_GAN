import logging


class Logger:
    def __init__(self, logfile):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s: - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S')

        # 使用FileHandler输出到文件
        fh = logging.FileHandler(logfile)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)

        # 使用StreamHandler输出到屏幕
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)

        # 添加两个Handler
        self.logger.addHandler(ch)
        self.logger.addHandler(fh)
    
    def print(self, msg):
        self.logger.info(msg)
