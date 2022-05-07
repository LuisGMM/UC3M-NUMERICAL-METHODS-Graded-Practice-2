import logging

logging.basicConfig(level=logging.INFO, filename='log.log', filemode='w',
                    format='%(asctime)s %(levelname)s %(message)s')

# Directly log with the basicConfig above

# logging.info('info')
# logging.debug('debug')
# logging.warning('warning')
# logging.error('error')
# logging.critical('critical')

# Create a custom logger to use in other modules

# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)

# handler = logging.FileHandler('log.log')
# formatter = logging.Formatter('%(asctime)s - %(levelname) - %(message)s')
# handler.setFormatter(formatter)

# logger.addHandler(handler)

# logger.info('info')
# logger.debug('debug')
# logger.warnning('warning')
# logger.error('error')
# logger.critical('critical')

# Two ways to log exceptions

# try:
#     1/0
# except ZeroDivisionError as e:
#     logging.error('ZeroDivisionError', exc_info=True)


# try:
#     1/0
# except ZeroDivisionError as e:
#     logging.exception('ZeroDivisionError')
