import logging
from typing import Optional

import pandas as pd

from src.api_client import KISClient
from src.models import AccountBalance, Position

logger = logging.getLogger("kis_trader.account")


class AccountAPI:
    """국내주식 계좌 조회 API."""

    def __init__(self, client: KISClient):
        self.client = client

    def get_balance(self) -> Optional[AccountBalance]:
        """주식 잔고를 조회한다 (보유종목 + 계좌 요약)."""
        params = {
            "CANO": "",
            "ACNT_PRDT_CD": "",
            "AFHR_FLPR_YN": "N",
            "OFL_YN": "",
            "INQR_DVSN": "02",  # 종목별
            "UNPR_DVSN": "01",
            "FUND_STTL_ICLD_YN": "N",
            "FNCG_AMT_AUTO_RDPT_YN": "N",
            "PRCS_DVSN": "00",
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": "",
        }

        res = self.client.get(
            api_url="/uapi/domestic-stock/v1/trading/inquire-balance",
            tr_id="TTTC8434R",
            params=params,
        )
        if not res.success:
            logger.error("잔고 조회 실패: %s", res.error_message)
            return None

        # 보유종목
        positions = []
        for item in (res.output1 or []):
            qty = int(item.get("hldg_qty", 0))
            if qty == 0:
                continue
            positions.append(Position(
                symbol=item.get("pdno", ""),
                name=item.get("prdt_name", ""),
                quantity=qty,
                avg_price=float(item.get("pchs_avg_pric", 0)),
                current_price=int(item.get("prpr", 0)),
                eval_amount=int(item.get("evlu_amt", 0)),
                profit_loss=int(item.get("evlu_pfls_amt", 0)),
                profit_rate=float(item.get("evlu_pfls_rt", 0)),
            ))

        # 계좌 요약 (output2의 첫 번째 항목)
        summary = {}
        if res.output2 and isinstance(res.output2, list) and len(res.output2) > 0:
            summary = res.output2[0]

        return AccountBalance(
            total_eval_amount=int(summary.get("tot_evlu_amt", 0)),
            total_deposit=int(summary.get("dnca_tot_amt", 0)),
            total_profit_loss=int(summary.get("evlu_pfls_smtl_amt", 0)),
            total_profit_rate=float(summary.get("tot_evlu_pfls_amt_rt", 0) or 0),
            positions=positions,
        )

    def get_buying_power(self, symbol: str = "", price: int = 0) -> int:
        """매수 가능 금액을 조회한다."""
        params = {
            "CANO": "",
            "ACNT_PRDT_CD": "",
            "PDNO": symbol,
            "ORD_UNPR": str(price),
            "ORD_DVSN": "01",  # 시장가
            "CMA_EVLU_AMT_ICLD_YN": "Y",
            "OVRS_ICLD_YN": "N",
        }

        res = self.client.get(
            api_url="/uapi/domestic-stock/v1/trading/inquire-psbl-order",
            tr_id="TTTC8908R",
            params=params,
        )
        if not res.success:
            logger.error("매수가능 조회 실패: %s", res.error_message)
            return 0

        output = res.output or {}
        return int(output.get("ord_psbl_cash", 0))

    def get_order_history(
        self,
        start_date: str,
        end_date: str,
        side: str = "00",
    ) -> pd.DataFrame:
        """주문 체결 내역을 조회한다.

        Args:
            start_date: 조회 시작일 (YYYYMMDD)
            end_date: 조회 종료일 (YYYYMMDD)
            side: "00":전체, "01":매도, "02":매수
        """
        params = {
            "CANO": "",
            "ACNT_PRDT_CD": "",
            "INQR_STRT_DT": start_date,
            "INQR_END_DT": end_date,
            "SLL_BUY_DVSN_CD": side,
            "PDNO": "",
            "CCLD_DVSN": "00",
            "INQR_DVSN": "00",
            "INQR_DVSN_3": "00",
            "ORD_GNO_BRNO": "",
            "ODNO": "",
            "INQR_DVSN_1": "",
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": "",
            "EXCG_ID_DVSN_CD": "KRX",
        }

        res = self.client.get(
            api_url="/uapi/domestic-stock/v1/trading/inquire-daily-ccld",
            tr_id="TTTC0081R",
            params=params,
        )
        if not res.success:
            logger.error("체결내역 조회 실패: %s", res.error_message)
            return pd.DataFrame()

        data = res.output1
        if not data:
            return pd.DataFrame()
        return pd.DataFrame(data)
