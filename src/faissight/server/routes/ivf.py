"""Inverted-list sizes and members."""

from __future__ import annotations

from typing import Annotated

import numpy as np
from fastapi import APIRouter, Query

from faissight.core import ivf
from faissight.server import schemas as S
from faissight.server.routes.common import (
    ApiError,
    IvfDep,
    snippet,
)

router = APIRouter()


@router.get("/ivf/lists", response_model=S.ListSizesResponse)
def ivf_lists(
    session: IvfDep, top: Annotated[int, Query(ge=1, le=1000)] = 20
) -> S.ListSizesResponse:
    st = session.list_stats()
    top_lists = [S.TopList(list_no=i, size=s) for i, s in st.top_lists[:top]]
    if top > len(st.top_lists):  # list_stats caches the default top-20
        order = np.argsort(-st.sizes, kind="stable")[:top]
        top_lists = [S.TopList(list_no=int(i), size=int(st.sizes[i])) for i in order]
    return S.ListSizesResponse(
        nlist=len(st.sizes),
        sizes=st.sizes.tolist(),
        n_empty=st.n_empty,
        imbalance_factor=st.imbalance_factor,
        min=st.min,
        median=st.median,
        max=st.max,
        top_lists=top_lists,
        top_5pct_share=st.top_5pct_share,
    )


@router.get("/ivf/list/{list_no}", response_model=S.ListMembersResponse)
def ivf_list(
    list_no: int,
    session: IvfDep,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=10_000)] = 100,
) -> S.ListMembersResponse:
    try:
        members = ivf.list_members(session.li, list_no)
    except IndexError as e:
        raise ApiError(404, "NOT_FOUND", str(e)) from e
    page = members[offset : offset + limit]
    return S.ListMembersResponse(
        list_no=list_no,
        size=len(members),
        offset=offset,
        limit=limit,
        members=[S.MemberOut(id=int(i), snippet=snippet(session, int(i))) for i in page],
    )
