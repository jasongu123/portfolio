"""
On-demand POSITION CHECKER / fill reconciliation for the live trend sleeve.

Run anytime (whether or not ibkr2.py is running) to see what the trend sleeve actually
HOLDS vs what it last INTENDED (trend_sleeve_state.json -> last_targets), and flag any
mismatch — unfilled orders, drift, or manual changes. Uses its OWN connection (clientId
91) so it never disturbs the running sleeve (clientId 90) or the equity client.

Usage:  python3.12 check_trend_positions.py 7497   # paper (or 7496 live)
"""

import os
import sys
import json
import threading
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from trend_sleeve_live import SYMBOLS, UNIVERSE, STATE_FILE

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 7497
CLIENT_ID = 92   # 91 is now taken by data_ingestion/ibkr_news.py (headline archiver)
BENIGN = {2104, 2106, 2107, 2108, 2158, 2100, 2150}


class PosClient(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.ok = threading.Event()
        self.done = threading.Event()
        self.pos = {}

    def nextValidId(self, oid):
        self.ok.set()

    def error(self, reqId, code, msg, misc=None):
        if code not in BENIGN:
            print(f"  [err {reqId}] {code}: {msg}")

    def position(self, account, contract, pos, avgCost):
        if contract.secType == 'STK':
            self.pos[contract.symbol] = (pos, avgCost)

    def positionEnd(self):
        self.done.set()


def main():
    app = PosClient()
    app.connect('127.0.0.1', PORT, CLIENT_ID)
    threading.Thread(target=app.run, daemon=True).start()
    if not app.ok.wait(8):
        sys.exit(f"Not connected on {PORT}. TWS/Gateway up & logged in? 7496 live / 7497 paper. "
                 f"If '326 client id in use', change CLIENT_ID.")
    app.reqPositions()
    app.done.wait(10)
    app.cancelPositions()

    targets, last_date = {}, None
    if os.path.exists(STATE_FILE):
        st = json.load(open(STATE_FILE))
        targets = st.get('last_targets', {}) or {}
        last_date = st.get('last_rebalance_date')

    print(f"\nTREND SLEEVE POSITION CHECK   (last rebalance: {last_date or 'NEVER'})")
    print(f"  {'sym':5s}{'sector':14s}{'current':>9s}{'target':>9s}{'diff':>8s}{'~notional':>13s}  flag")
    gross = net = 0.0
    mismatches = held = 0
    for s in SYMBOLS:
        cur, avg = app.pos.get(s, (0, 0.0))
        cur = int(cur)
        tgt = int(targets.get(s, 0))
        diff = tgt - cur
        notl = cur * avg
        gross += abs(notl)
        net += notl
        if cur:
            held += 1
        if s in targets:
            flag = 'OK' if diff == 0 else f'UNFILLED/DRIFT {diff:+d}'
            if diff != 0:
                mismatches += 1
        else:
            flag = '(no target)'
        print(f"  {s:5s}{UNIVERSE.get(s,''):14s}{cur:>9d}{tgt:>9d}{diff:>8d}{notl:>13,.0f}  {flag}")
    app.disconnect()

    print(f"\n  held: {held}/{len(SYMBOLS)} | gross ${gross:,.0f} | net ${net:,.0f} "
          f"| mismatches vs last target: {mismatches}")
    if not targets:
        print("  (no saved targets yet — the sleeve hasn't rebalanced. 'current' is whatever's in the account.)")
    elif mismatches:
        print("  -> mismatches = orders that didn't fully fill, or positions that drifted. "
              "Re-run after a minute; if persistent, check TWS order status.")
    else:
        print("  -> all positions match the sleeve's last targets. Clean.")


if __name__ == '__main__':
    main()
