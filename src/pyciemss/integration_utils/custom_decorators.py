import logging
import time
import functools

def pyciemss_logging_wrapper( function):
    def wrapped(*args, **kwargs):
        try:
            start_time = time.perf_counter()
            result = function(*args, **kwargs)
            end_time = time.perf_counter()
            logging.info(
                "Elapsed time for %s: %f",
                function.__name__, end_time - start_time
            )
            return result
        except Exception as e:

            log_message = """
                ###############################

                There was an exception in pyciemss
                
                Error occured in function: %s

                Function docs : %s

                ################################
            """
            logging.exception(log_message, function.__name__, function.__doc__)
            raise e
    functools.update_wrapper(wrapped, function)
    return wrapped
