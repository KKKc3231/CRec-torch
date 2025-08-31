import logging

def setup_logger(name, log_file=None, level=logging.INFO):

    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        
        # 创建文件处理器（如果指定了日志文件）
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
        
        # 创建格式化器并添加到处理器
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        if log_file:
            file_handler.setFormatter(formatter)
        
        # 将处理器添加到日志记录器
        logger.addHandler(console_handler)
        if log_file:
            logger.addHandler(file_handler)
    
    return logger