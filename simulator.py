
class POSITION_TYPE:
  SHORT = -1
  LONG = 1
  NONE = 0

class Position:
  def __init__(self, ask, bid, type, time, symbol):
    self.type = type
    self.openask = ask
    self.openbid = bid
    self.symbol = symbol
    self.closed = False

  def close(self, ask, bid, time):
    self.closeask = ask
    self.closebid = bid
    self.equity = (self.openbid - ask) if self.type == POSITION_TYPE.SHORT else (bid - self.openask)
    self.closed = True
  
  def set_stop_loss(self, value):
    self.stop_loss = value

  def set_take_profit(self, value):
    self.take_profit = value 

  def tick(self, ask, bid, time):
    if ( ( self.type == POSITION_TYPE.SHORT ) and ( ( bid > self.stop_loss) or ( bid < self.take_profit ) ) ) or ( ( self.type == POSITION_TYPE.LONG ) and ( ( ask < self.stop_loss) or ( ask > self.take_profit ) ) ):
      self.close(ask, bid, time)

  def get_equity(self):
    return self.equity