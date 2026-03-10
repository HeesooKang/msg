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

    @staticmethod
    def _coerce_int(value) -> Optional[int]:
        """API 문자열/숫자 값을 int로 변환한다."""
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return int(value)

        text = str(value).strip().replace(",", "")
        if not text:
            return None

        try:
            return int(float(text))
        except ValueError:
            return None

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
            config = getattr(self.client, "config", None)
            acc_no = getattr(config, "account_number", "")
            acc_prod = getattr(config, "account_product_code", "")
            if acc_no:
                acc_masked = f"{acc_no[:2]}******" if len(acc_no) > 2 else "***"
            else:
                acc_masked = "***"

            if res.error_code == "OPSQ2000" and "INVALID_CHECK_ACNO" in res.error_message:
                logger.error(
                    "잔고 조회 실패: INVALID_CHECK_ACNO (입력값 체크 실패) "
                    "CANO=%s, ACNT_PRDT_CD=%s, MODE=%s",
                    acc_masked,
                    acc_prod,
                    getattr(config, "trading_mode", "unknown"),
                )
            else:
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

    def get_realized_profit_loss(self) -> Optional[int]:
        """당일 실현손익(원)을 조회한다.

        주식잔고조회_실현손익(v1_국내주식-041) 기준이며,
        전일 매매를 제외(PRCS_DVSN=01)한 당일 값을 사용한다.
        """
        params = {
            "CANO": "",
            "ACNT_PRDT_CD": "",
            "AFHR_FLPR_YN": "N",
            "OFL_YN": "",
            "INQR_DVSN": "02",
            "UNPR_DVSN": "01",
            "FUND_STTL_ICLD_YN": "N",
            "FNCG_AMT_AUTO_RDPT_YN": "N",
            "PRCS_DVSN": "01",  # 전일매매 미포함 (당일 기준)
            "COST_ICLD_YN": "N",
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": "",
        }

        res = self.client.get(
            api_url="/uapi/domestic-stock/v1/trading/inquire-balance-rlz-pl",
            tr_id="TTTC8494R",
            params=params,
        )
        if not res.success:
            logger.warning("실현손익 조회 실패: %s", res.error_message)
            return None

        # output2 요약값 우선
        if isinstance(res.output2, list) and res.output2:
            summary = res.output2[0] if isinstance(res.output2[0], dict) else {}
            for key in ("rlzt_pfls", "real_evlu_pfls"):
                parsed = self._coerce_int(summary.get(key))
                if parsed is not None:
                    return parsed

        # output1 종목별 실현손익 합산 fallback
        if isinstance(res.output1, list):
            total = 0
            found = False
            for item in res.output1:
                if not isinstance(item, dict):
                    continue
                parsed = self._coerce_int(item.get("rlzt_pfls"))
                if parsed is None:
                    continue
                total += parsed
                found = True
            if found:
                return total

        logger.warning("실현손익 응답 파싱 실패: rlzt_pfls 값을 찾지 못했습니다.")
        return None

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
