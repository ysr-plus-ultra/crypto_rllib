from binance import ThreadedWebsocketManager
import pymongo
from binance.enums import HistoricalKlinesType
from tqdm import tqdm
from binance.client import Client
api_key = "k3981gJNtE8gxkWtXMGDvtW62nLeiDVgb0qLSggOXVVFbT4ks50j3XKvXCbc3jMI"
api_secret = "TmiJjVZ6DNWjhvnZlP1y03yGoBtYUvVd7nfJ9I2I7gXUl0JbewSmTxYGPN4F5Xea"
client = Client(api_key, api_secret)
mongo_client = pymongo.MongoClient("mongodb://ysr1004:q5n76hrh@192.168.0.11:27017")
streams = ['btcusdt@kline_1m', 'ethusdt@kline_1m', 'solusdt@kline_1m', 'bnbusdt@kline_1m', 'adausdt@kline_1m']
symbols = ['ADAUSDT', 'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT']
db = mongo_client['BINANCE_FUTURES']
def main():
    for market_type in [HistoricalKlinesType.FUTURES]:
        for symbol in symbols:
            start_str = "1 days ago UTC"
            for x in tqdm(client.get_historical_klines_generator(symbol=symbol,
                                                                 interval=client.KLINE_INTERVAL_1MINUTE,
                                                                 start_str=start_str,
                                                                 klines_type=market_type), smoothing=0.9):
                collection = db['binance_' + symbol + '_1m']
                collection.update_one(
                    {"_id": x[0]}, {
                        "$set": {"Open": float(x[1]),
                                 "High": float(x[2]),
                                 "Low": float(x[3]),
                                 "Close": float(x[4]),
                                 "Volume": float(x[5]),
                                 }}, upsert=True)
    twm = ThreadedWebsocketManager(api_key=api_key, api_secret=api_secret)
    # start is required to initialise its internal loop
    twm.daemon = True
    twm.start()

    def handle_socket_message(msg):
        db = mongo_client.BINANCE_FUTURES
        collection = db['binance_'+msg['data']['s']+'_'+msg['data']['k']['i']]
        collection.update_one(
            {"_id": msg["data"]["k"]['t']}, {
                "$set": {"Open": float(msg["data"]["k"]["o"]),
                         "Close": float(msg["data"]["k"]["c"]),
                         "High": float(msg["data"]["k"]["h"]),
                         "Low": float(msg["data"]["k"]["l"]),
                         "Volume": float(msg["data"]["k"]["n"]),
                         }}, upsert=True)

    twm.start_multiplex_socket(callback=handle_socket_message, streams=streams)

    twm.join()


if __name__ == "__main__":
    main()