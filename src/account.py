import logging
import time
from typing import Optional

import pandas as pd

from src.api_client import KISClient
from src.models import AccountBalance, Position

logger = logging.getLogger("kis_trader.account")


class AccountAPI:
    """국내주식 계좌 조회 API."""

    def __init__(self, client: KISClient):
        self.client = client

    def _get_all_pages(
        self,
        *,
        api_url: str,
        tr_id: str,
        params: dict,
        max_pages: int = 20,
    ) -> list:
        responses = []
        page_params = dict(params or {})
        tr_cont = ""

        for _ in range(max_pages):
            res = self.client.get(
                api_url=api_url,
                tr_id=tr_id,
                params=page_params,
                tr_cont=tr_cont,
                log_timeout=False,
            )
            if not res.success:
                return responses + [res]
            responses.append(res)
            if not res.has_next:
                return responses

            payload = res.data or {}
            next_fk = str(payload.get("ctx_area_fk100", "") or "")
            next_nk = str(payload.get("ctx_area_nk100", "") or "")
            if not next_fk and not next_nk:
                logger.warning("연속조회 헤더가 남아 있지만 다음 페이지 키가 없어 중단합니다: %s", api_url)
                return responses

            page_params["CTX_AREA_FK100"] = next_fk
            page_params["CTX_AREA_NK100"] = next_nk
            tr_cont = "N"
            time.sleep(0.1)

        logger.warning("연속조회 최대 페이지(%d)를 넘어 중단합니다: %s", max_pages, api_url)
        return responses

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

        responses = self._get_all_pages(
            api_url="/uapi/domestic-stock/v1/trading/inquire-balance",
            tr_id="TTTC8434R",
            params=params,
        )
        if not responses:
            return None
        res = responses[-1]
        if not res.success:
            config = self.client.config
            acc_no = config.account_number
            acc_prod = config.account_product_code
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
                    config.trading_mode,
                )
            else:
                logger.error("잔고 조회 실패: %s", res.error_message)
            return None

        # 보유종목
        positions = []
        for page in responses:
            for item in (page.output1 or []):
                qty = int(item.get("hldg_qty", 0))
                if qty == 0:
                    continue
                positions.append(Position(
                    symbol=item.get("pdno", ""),
                    quantity=qty,
                    avg_price=float(item.get("pchs_avg_pric", 0)),
                ))

        # 계좌 요약 (output2의 첫 번째 항목)
        summary = {}
        for page in responses:
            if page.output2 and isinstance(page.output2, list) and len(page.output2) > 0:
                summary = page.output2[0]

        return AccountBalance(
            total_eval_amount=int(summary.get("tot_evlu_amt", 0)),
            total_deposit=int(summary.get("dnca_tot_amt", 0)),
            total_profit_loss=int(summary.get("evlu_pfls_smtl_amt", 0)),
            positions=positions,
        )

    def get_realized_profit_loss(self) -> Optional[int]:
        """당일 실현손익(원)을 조회한다.

        주식잔고조회_실현손익(v1_국내주식-041) 기준이며,
        전일 매매를 제외(PRCS_DVSN=01)한 당일 값을 사용한다.
        """
        if self.client.config.is_paper:
            return None

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

        responses = self._get_all_pages(
            api_url="/uapi/domestic-stock/v1/trading/inquire-balance-rlz-pl",
            tr_id="TTTC8494R",
            params=params,
        )
        if not responses:
            return None
        res = responses[-1]
        if not res.success:
            logger.warning("실현손익 조회 실패: %s", res.error_message)
            return None

        # output2 요약값 우선
        for page in responses:
            if isinstance(page.output2, list) and page.output2:
                summary = page.output2[0] if isinstance(page.output2[0], dict) else {}
                for key in ("rlzt_pfls", "real_evlu_pfls"):
                    parsed = self._coerce_int(summary.get(key))
                    if parsed is not None:
                        return parsed

        # output1 종목별 실현손익 합산 fallback
        found = False
        total = 0
        for page in responses:
            if not isinstance(page.output1, list):
                continue
            for item in page.output1:
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

    def get_order_history(
        self,
        start_date: str,
        end_date: str,
        side: str = "00",
        *,
        symbol: str = "",
        order_no: str = "",
    ) -> pd.DataFrame:
        """주문 체결 내역을 조회한다.

        Args:
            start_date: 조회 시작일 (YYYYMMDD)
            end_date: 조회 종료일 (YYYYMMDD)
            side: "00":전체, "01":매도, "02":매수
            symbol: 종목코드 필터
            order_no: 주문번호 필터
        """
        params = {
            "CANO": "",
            "ACNT_PRDT_CD": "",
            "INQR_STRT_DT": start_date,
            "INQR_END_DT": end_date,
            "SLL_BUY_DVSN_CD": side,
            "PDNO": str(symbol or "").strip(),
            "CCLD_DVSN": "00",
            "INQR_DVSN": "00",
            "INQR_DVSN_3": "00",
            "ORD_GNO_BRNO": "",
            "ODNO": str(order_no or "").strip(),
            "INQR_DVSN_1": "",
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": "",
            "EXCG_ID_DVSN_CD": "KRX",
        }

        responses = self._get_all_pages(
            api_url="/uapi/domestic-stock/v1/trading/inquire-daily-ccld",
            tr_id="TTTC0081R",
            params=params,
        )
        if not responses:
            return pd.DataFrame()
        res = responses[-1]
        if not res.success:
            logger.error("체결내역 조회 실패: %s", res.error_message)
            return pd.DataFrame()

        data = []
        for page in responses:
            if page.output1:
                data.extend(page.output1)
        if not data:
            return pd.DataFrame()
        return pd.DataFrame(data)
