"""Tests for accessory alignment updates from location reports."""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta, timezone

import pytest
from typing_extensions import override

from findmy import FindMyAccessory, KeyPair
from findmy.reports.reports import LocationReport, LocationReportsFetcher, _key_index_bounds

INTERVAL = timedelta(minutes=15)
# LocationReport timestamps count seconds from 2001-01-01.
APPLE_EPOCH_OFFSET = 60 * 60 * 24 * 11323


def _accessory(index: int, at: datetime) -> FindMyAccessory:
    return FindMyAccessory(
        master_key=KeyPair.new().private_key_bytes,
        skn=os.urandom(32),
        sks=os.urandom(32),
        paired_at=at - index * INTERVAL,
        alignment_date=at,
        alignment_index=index,
    )


class _FakeReport(LocationReport):
    """Report for a key at a given time, without an encrypted location."""

    def __init__(self, key: KeyPair, timestamp: datetime) -> None:
        payload = (int(timestamp.timestamp()) - APPLE_EPOCH_OFFSET).to_bytes(4, "big")
        super().__init__(payload, key.hashed_adv_key_bytes)

    @override
    def decrypt(self, key: KeyPair) -> None:
        self._decrypted_data = (key, bytes(10))


class _FakeAccount:
    """Account that returns a report for every requested key in `reports`."""

    def __init__(self, reports: dict[str, LocationReport]) -> None:
        self._reports = reports

    async def fetch_raw_reports(
        self,
        devices: list[tuple[list[str], list[str]]],
    ) -> list[LocationReport]:
        return [
            self._reports[key]
            for primary, secondary in devices
            for key in (*primary, *secondary)
            if key in self._reports
        ]


@pytest.mark.parametrize("offset", [0, 1, 2])
def test_secondary_key_index_bounds(offset: int) -> None:
    """A secondary key spans every primary index that may use it, not only the generated ones."""
    acc = _accessory(index=10_000, at=datetime.now(tz=timezone.utc))
    secondary = 10_000 // 96 + 2
    key = acc._secondary_gen[secondary]  # noqa: SLF001

    # only part of the range is known, e.g. because keys were generated in a batch
    known = {(secondary - 1) * 96 + offset}
    assert _key_index_bounds(acc, key, known) == ((secondary - 2) * 96, secondary * 96 - 1)


def test_primary_key_index_bounds() -> None:
    """A primary key belongs to a single index."""
    acc = _accessory(index=10_000, at=datetime.now(tz=timezone.utc))
    key = acc._primary_gen[10_005]  # noqa: SLF001

    assert _key_index_bounds(acc, key, {10_005}) == (10_005, 10_005)


def test_secondary_reports_do_not_move_alignment_ahead(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Fetches every 15 minutes that keep finding the current secondary key keep the tag in range.

    The accessory rolls its primary index every 15 minutes and stays on one secondary key, which is
    shared by indices up to two days ahead of the primary one. Reports arrive two hours late.
    """
    start = datetime(2026, 10, 3, 9, 0, tzinfo=timezone.utc)
    first_index = 10_000 * 96 + 20  # early in a secondary key period
    acc = _accessory(index=first_index, at=start)
    secondary_key = acc._secondary_gen[first_index // 96 + 2]  # noqa: SLF001

    clock = start

    class _Clock(datetime):
        @classmethod
        @override
        def now(cls, tz: timezone | None = None) -> datetime:  # pyright: ignore[reportIncompatibleMethodOverride]
            return clock if tz is None else clock.astimezone(tz)

    monkeypatch.setattr("findmy.reports.reports.datetime", _Clock)

    for fetch in range(1, 11):
        clock = start + fetch * INTERVAL
        current = first_index + fetch

        report = _FakeReport(secondary_key, clock - timedelta(hours=2))
        account = _FakeAccount({secondary_key.hashed_adv_key_b64: report})
        fetcher = LocationReportsFetcher(account)  # pyright: ignore[reportArgumentType]

        reports = asyncio.run(fetcher._fetch_accessory_reports(acc, only_latest=True))  # noqa: SLF001
        assert reports == [report]

        assert acc.get_min_index(clock) <= current <= acc.get_max_index(clock)
        # local matching only has to look a short way around the accessory's alignment
        assert acc.get_max_index(clock) - current <= 2 * 96
