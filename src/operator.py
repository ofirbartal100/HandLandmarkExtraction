class Operator():
    def __init__(self, input_handler, processer, output_handler, data):
        self.input_handler = input_handler
        self.processer = processer
        self.output_handler = output_handler
        self.data = data

    def operate(self):
        input = self.input_handler.handle(self.data)
        output = self.processer.process(input)
        self.output_handler(output)
