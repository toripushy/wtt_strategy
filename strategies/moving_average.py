class Strategy:
    frequency = "1m"

    def __init__(self):
        self.fast = 5
        self.slow = 20

    def on_bar(self, ctx, data):
        for symbol in data:
            fast = ctx.history(symbol, "close", self.fast)
            slow = ctx.history(symbol, "close", self.slow)
            if len(slow) < self.slow:
                continue
            if sum(fast) / len(fast) > sum(slow) / len(slow):
                ctx.order_target_percent(symbol, 0.95)
            else:
                ctx.order_target_percent(symbol, 0.0)
