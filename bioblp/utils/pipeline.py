

def create_pipeline(functions: list):
    """Sequentially executes a list of functions"""
    def pipeline(input):
        res = input
        for function in functions:
            res = function(res)
        return res

    return pipeline
