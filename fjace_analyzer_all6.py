#!/usr/bin/env python3
"""FJACE Advanced Statistical Analyzer for Janggi.
[ 단일/다중 기보 하이브리드 AI 유사도 프로파일링 도구 - v4.3 ]
- 기존 v3.0 코어 로직 및 UI (가이드라인, 합의 패널, 스파이크 감지, 핑거프린팅 등) 100% 보존
- DB 연동: SQLite 기반 유저 프로파일링 및 환경(Depth/Nodes) 격리형 저장
- 서술형 플레이 성향 리포트 제공 (공격적/수비적/기복 분석)
- [고도화] 4단계 피드백 라벨링(AI확신~사람확신) 전면 수학적 반영 (K-NN 다중 군집 분석)
"""

from __future__ import annotations

import argparse
import math
import re
import subprocess
import sys
import csv
import time
import sqlite3
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# ==============================================================================
# DB 연동 및 성향 분석 / 라벨링 모듈 (Depth/Node 격리형)
# ==============================================================================
DB_FILE = "janggi_profiles.db"

def get_search_config(depth: Optional[int], nodes: Optional[int]) -> str:
    if nodes: return f"N{nodes}"
    return f"D{depth}"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS players_v2 (
            nickname TEXT,
            search_config TEXT,
            total_games INTEGER DEFAULT 0,
            avg_acpl REAL DEFAULT 0,
            avg_top1 REAL DEFAULT 0,
            avg_blunder REAL DEFAULT 0,
            avg_std_dev REAL DEFAULT 0,
            primary_opening TEXT DEFAULT '분석중',
            playstyle TEXT DEFAULT '분석중',
            last_played TIMESTAMP,
            PRIMARY KEY (nickname, search_config)
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS labeled_stats_v2 (
            label_id INTEGER,
            search_config TEXT,
            label_name TEXT,
            count INTEGER DEFAULT 0,
            avg_acpl REAL DEFAULT 0.0,
            avg_top1 REAL DEFAULT 0.0,
            avg_std_dev REAL DEFAULT 0.0,
            PRIMARY KEY (label_id, search_config)
        )
    ''')
    conn.commit()
    conn.close()

def detect_opening(uci_moves: List[str]) -> str:
    if len(uci_moves) < 2: return "기타/알수없음"
    moves_str = ",".join(uci_moves[:3])
    if "b0c2" in moves_str or "h0g2" in moves_str or "b9c7" in moves_str or "h9g7" in moves_str:
        if "c0e2" in moves_str or "g0e2" in moves_str or "c9e7" in moves_str or "g9e7" in moves_str:
            return "면상 (Myun-Sang)"
        return "귀마 (Gwi-ma)"
    elif "b0c2" in moves_str and "h0g2" in moves_str: return "원앙마 (Won-ang-ma)"
    elif "b0d2" in moves_str or "h0f2" in moves_str: return "양귀마 (Yang-gwi-ma)"
    return "변칙/동네장기"

def update_player_profile(nickname: str, stats: tuple, uci_moves: List[str], search_config: str):
    if not nickname or nickname == "-" or nickname.lower().startswith("player"): return
    acpl, top1, top3, perf, inacc, mist, blunder, std_dev, crit_top1, u_top1, u_acpl = stats
    
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT * FROM players_v2 WHERE nickname = ? AND search_config = ?", (nickname, search_config))
    row = c.fetchone()
    now = datetime.datetime.now()
    opening = detect_opening(uci_moves)
    
    if row is None:
        style = "안정적 수비형" if std_dev < 25 else "난전/기복형"
        c.execute('''INSERT INTO players_v2 (nickname, search_config, total_games, avg_acpl, avg_top1, avg_blunder, avg_std_dev, primary_opening, playstyle, last_played)
                     VALUES (?, ?, 1, ?, ?, ?, ?, ?, ?, ?)''', (nickname, search_config, u_acpl, u_top1, blunder, std_dev, opening, style, now))
        print(f" [DB 저장 완료] '{nickname}'의 신규 프로필이 생성되었습니다. (기준: {search_config})")
    else:
        tg = row[2] + 1
        new_acpl = ((row[3] * row[2]) + u_acpl) / tg
        new_top1 = ((row[4] * row[2]) + u_top1) / tg
        new_blun = ((row[5] * row[2]) + blunder) / tg
        new_std  = ((row[6] * row[2]) + std_dev) / tg
        
        style = "다재다능"
        if new_std > 35 and new_blun > 2: style = "공격/기복형 (Aggressive/Inconsistent)"
        elif new_std < 22 and new_blun < 1: style = "견고한 수비형 (Solid/Defensive)"
        elif new_top1 > 55 and new_acpl < 25: style = "정통파 고수 (Orthodox Expert)"
        
        c.execute('''UPDATE players_v2 SET total_games=?, avg_acpl=?, avg_top1=?, avg_blunder=?, avg_std_dev=?, playstyle=?, last_played=? 
                     WHERE nickname=? AND search_config=?''', 
                  (tg, new_acpl, new_top1, new_blun, new_std, style, now, nickname, search_config))
        print(f" [DB 저장 완료] '{nickname}'의 누적 프로필이 업데이트 되었습니다. (총 {tg}판 / 기준: {search_config})")
    conn.commit()
    conn.close()

def update_labeled_stats(label_id: int, label_name: str, stats: tuple, search_config: str):
    acpl, top1, top3, perf, inacc, mist, blunder, std_dev, crit_top1, u_top1, u_acpl = stats
    
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT count, avg_acpl, avg_top1, avg_std_dev FROM labeled_stats_v2 WHERE label_id = ? AND search_config = ?", (label_id, search_config))
    row = c.fetchone()
    
    if row:
        cnt, a_acpl, a_top1, a_std = row
        new_cnt = cnt + 1
        new_acpl = ((a_acpl * cnt) + u_acpl) / new_cnt
        new_top1 = ((a_top1 * cnt) + u_top1) / new_cnt
        new_std = ((a_std * cnt) + std_dev) / new_cnt
        c.execute("UPDATE labeled_stats_v2 SET count=?, avg_acpl=?, avg_top1=?, avg_std_dev=? WHERE label_id=? AND search_config=?", 
                  (new_cnt, new_acpl, new_top1, new_std, label_id, search_config))
        print(f" [DB 라벨링 누적] '{label_name}' 통계가 갱신되었습니다. (총 {new_cnt}건 / 기준: {search_config})")
    else:
        c.execute('''INSERT INTO labeled_stats_v2 (label_id, search_config, label_name, count, avg_acpl, avg_top1, avg_std_dev)
                     VALUES (?, ?, ?, 1, ?, ?, ?)''', (label_id, search_config, label_name, u_acpl, u_top1, std_dev))
        print(f" [DB 라벨링 생성] '{label_name}' 그룹이 신규 생성되었습니다. (기준: {search_config})")
        
    conn.commit()
    conn.close()

def extract_player_names(pgn_text: str) -> Tuple[str, str]:
    w_match = re.search(r'\[White "([^"]+)"\]', pgn_text)
    b_match = re.search(r'\[Black "([^"]+)"\]', pgn_text)
    white = w_match.group(1) if w_match and w_match.group(1) != "-" else ""
    black = b_match.group(1) if b_match and b_match.group(1) != "-" else ""
    return white, black

def calibrate_with_db(nickname: str, base_prob: float, base_part_prob: float, stats: tuple, use_db: bool, search_config: str) -> Tuple[float, float, str]:
    """
    [V10.0 정밀 군집 엔진]
    블랙홀 현상 수정: 거리 허용치를 엄격하게 좁혀, 
    정말로 패턴이 일치할 때만(ACPL 오차 10 이내, 편차 오차 15 이내) DB가 확률을 덮어씌웁니다.
    """
    if not use_db: return base_prob, base_part_prob, ""
    
    acpl, top1, top3, perf, inacc, mist, blunder, std_dev, crit_top1, u_top1, u_acpl = stats
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    new_prob = base_prob
    new_part_prob = base_part_prob
    msg = ""

    c.execute("SELECT label_id, label_name, count, avg_acpl, avg_top1, avg_std_dev FROM labeled_stats_v2 WHERE count >= 1 AND search_config=?", (search_config,))
    labels = c.fetchall()
    
    db_influence = 0.0
    
    if labels:
        # [핵심 1] 단위를 엄격하게 쪼임 (사람과 AI를 가르는 벽)
        # ACPL 10 차이, Top-1 20% 차이, 편차 20 차이를 '완전히 다른 차원(거리 1.0)'으로 인식함.
        def calc_distance(a_acpl, a_top1, a_std, b_acpl, b_top1, b_std):
            d_acpl = abs(a_acpl - b_acpl) / 10.0    
            d_top1 = abs(a_top1 - b_top1) / 20.0    
            d_std = abs(a_std - b_std) / 20.0       
            return math.sqrt(d_acpl**2 + d_top1**2 + d_std**2)

        min_dist = 9999.0
        closest_label_id = None
        closest_name = ""
        total_labeled_games = sum(row[2] for row in labels)

        for l_id, l_name, l_count, l_acpl, l_top1, l_std in labels:
            dist = calc_distance(u_acpl, u_top1, std_dev, l_acpl, l_top1, l_std)
            if dist < min_dist:
                min_dist = dist
                closest_label_id = l_id
                closest_name = l_name

        # [핵심 2] 거리가 1.5 이상 멀어지면 DB 영향력 0%로 차단 (블랙홀 현상 방지)
        # 거리가 가까울수록 영향력이 급격히 100%에 도달하도록 조정
        data_weight = min(1.0, total_labeled_games / 3.0) 
        dist_weight = max(0.0, 1.0 - (min_dist / 1.5)) 
        
        db_influence = math.sqrt(data_weight * dist_weight)

        target_probs = {
            1: 95.0,  # AI 확신
            2: 75.0,  # AI 의심
            3: 30.0,  # 사람 추정
            4:  5.0   # 사람 확신
        }
        
        if closest_label_id in target_probs and db_influence > 0.05:
            db_target = target_probs[closest_label_id]
            
            # 수학 공식(base_prob)을 DB값(db_target)으로 강력하게 보정
            new_prob = (base_prob * (1.0 - db_influence)) + (db_target * db_influence)
            
            # 부분 스파이크 억제 및 보정
            if closest_label_id in (3, 4): 
                new_part_prob = (base_part_prob * (1.0 - db_influence)) + (target_probs[closest_label_id] * db_influence)
            elif closest_label_id == 1:
                new_part_prob = max(base_part_prob, (base_part_prob * (1.0 - db_influence)) + (95.0 * db_influence))

            msg += f" [DB패턴: '{closest_name}'에 일치 (신뢰도 {db_influence*100:.0f}% 적용)]"
        else:
            # 패턴에 맞지 않을 경우 사용자에게 안내
            if min_dist >= 1.5:
                msg += f" [DB: 기존 라벨링 패턴과 거리가 멀어 수학 공식을 유지함]"

    # 개인 닉네임 DB 보정 (유지)
    if nickname and nickname != "-" and not nickname.lower().startswith("player"):
        c.execute("SELECT total_games, avg_acpl, avg_std_dev FROM players_v2 WHERE nickname=? AND search_config=?", (nickname, search_config))
        row = c.fetchone()
        if row and row[0] >= 3: 
            hist_acpl = row[1]
            if hist_acpl - u_acpl > 15 and new_prob > 50:
                new_prob = min(99.9, new_prob + 10.0)
                new_part_prob = min(99.9, new_part_prob + 10.0)
                msg += f" [!개인DB: 평소 실력 대비 극단적 성능 급등]"

    conn.close()
    return min(99.9, max(0.0, new_prob)), min(99.9, max(0.0, new_part_prob)), msg

def get_playstyle_narrative(nickname: str, search_config: str, current_moves: List[HistoryEntry]) -> str:
    """DB에 누적된 성향 지표를 바탕으로 플레이 스타일을 서술형 텍스트로 풀어냅니다."""
    if not nickname or nickname == "-" or nickname.lower().startswith("player"):
        stats = calc_stats(current_moves)
        std_dev, blunder = stats[7], stats[6]
        narrative = " 닉네임이 지정되지 않아 금일 1판의 대국에 국한된 단기 성향입니다.\n"
        if std_dev > 35 and blunder >= 2:
            narrative += "   👉 이 대국에서는 큰 난전과 전투가 벌어졌으며, 공격적이나 기복과 치명적 실수가 관찰되는 다이나믹한 흐름이었습니다."
        elif std_dev < 20 and blunder == 0:
            narrative += "   👉 이 대국에서는 수비적이고 매우 단단한 수읽기를 통해 기복 없이 안정적인(Solid) 방어 성향을 보여주었습니다."
        else:
            narrative += "   👉 전반적으로 공격과 수비의 밸런스가 맞춰진 일반적인 대국 전개 성향을 보여주었습니다."
        return narrative

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT total_games, avg_acpl, avg_top1, avg_blunder, avg_std_dev, primary_opening, playstyle FROM players_v2 WHERE nickname=? AND search_config=?", (nickname, search_config))
    row = c.fetchone()
    conn.close()

    if not row: return " 아직 DB에 데이터가 충분히 쌓이지 않아 상세 성향 파악이 대기 중입니다."
    
    t_games, a_acpl, a_top1, a_blun, a_std, opening, style = row
    
    reliability = "신뢰도 낮음"
    if t_games >= 10: reliability = "신뢰도 매우 높음 (성향 뚜렷함)"
    elif t_games >= 3: reliability = "신뢰도 보통 (데이터 누적 중)"

    narrative = f" 총 {t_games}판의 누적 데이터를 기반으로 분석한 결과 ({reliability}):\n"
    narrative += f"   👉 주로 [{opening}] 포진을 선호하며, 전반적인 대국 흐름은 [{style}]에 가깝습니다.\n"
    
    if a_std < 22:
        narrative += f"   👉 평균 기복(편차)이 {a_std:.1f}로 매우 낮아, 변칙적인 공격보다는 튼튼하게 진영을 갖추는 방어적이고 안정적인 운영을 선호합니다. "
    elif a_std > 35:
        narrative += f"   👉 평균 기복(편차)이 {a_std:.1f}로 높은 편으로, 수비보다는 난전을 유도하여 승부를 보는 공격적 성향이 강합니다. "
    else:
        narrative += f"   👉 평균 기복(편차)이 {a_std:.1f} 수준으로, 공수의 밸런스를 유연하게 가져가는 다재다능한 스타일입니다. "

    if a_blun < 0.5:
        narrative += f"특히 매 판 치명적 실수(Blunder)가 평균 {a_blun:.1f}회에 불과할 정도로 꼼꼼한 수읽기를 자랑합니다."
    elif a_blun > 2.0:
        narrative += f"다만 난전 속에서 치명적 실수(Blunder)가 평균 {a_blun:.1f}회씩 발생하여, 후반 집중력 보완이 필요해 보입니다."
        
    return narrative

# ==============================================================================
# 기존 v3.0 코어 로직 완벽 보존 구역 (JanggiBoard, EngineSession, Analyze 등)
# ==============================================================================
FILES = "abcdefghi"
BOARD_FILES = range(9)
BOARD_RANKS = range(10)

MOVE_RE = re.compile(r"^([RHEACKP]?)([a-i0-9]{0,2})([a-i])(10|[0-9])$")

@dataclass
class EngineMoveInfo:
    uci: str
    score: int

@dataclass
class HistoryEntry:
    ply: int
    san: str
    uci: str
    best_move: str
    top_3_moves: List[str]
    is_top_1: bool
    is_top_3: bool
    eval_loss: int
    current_eval: int
    is_critical: bool
    is_forced: bool
    eval_gap: int
    start_fen: Optional[str] = None
    uci_history: List[str] = None
    game_id: int = 0

@dataclass
class TacticalBurstResult:
    score: float
    start_ply: int
    end_ply: int
    game_id: int
    move_count: int
    acpl: float
    rank_match: float
    top1_rate: float
    std_dev: float
    baseline_acpl: float
    baseline_rank: float
    hard_count: int
    swing_count: int
    recovery_count: int
    commentary: str

def to_fs_uci(uci_0: str) -> str:
    m = re.match(r"^([a-i])(\d+)([a-i])(\d+)$", uci_0)
    if m: return f"{m.group(1)}{int(m.group(2))+1}{m.group(3)}{int(m.group(4))+1}"
    return uci_0

def from_fs_uci(uci_fs: str) -> str:
    m = re.match(r"^([a-i])(\d+)([a-i])(\d+)$", uci_fs)
    if m: return f"{m.group(1)}{int(m.group(2))-1}{m.group(3)}{int(m.group(4))-1}"
    return uci_fs

def cap_score(score: int) -> int:
    return max(-3000, min(3000, score))

@dataclass(frozen=True)
class Piece:
    color: str
    kind: str

class JanggiBoard:
    def __init__(self) -> None:
        self.board: Dict[Tuple[int, int], Piece] = {}
        self._initial_color = "w"
        self._setup_initial_position()

    def load_fen(self, fen: str) -> None:
        self.board.clear()
        parts = fen.split()
        rows = parts[0].split('/')
        for rank_idx, row in enumerate(rows):
            rank = 9 - rank_idx
            file_idx = 0
            for char in row:
                if char.isdigit(): file_idx += int(char)
                else:
                    self.board[(file_idx, rank)] = Piece("w" if char.isupper() else "b", char.upper())
                    file_idx += 1
        if len(parts) > 1: self._initial_color = parts[1]

    @staticmethod
    def parse_square(text: str) -> Tuple[int, int]: return FILES.index(text[0].lower()), int(text[1:])
    @staticmethod
    def in_bounds(square: Tuple[int, int]) -> bool: return square[0] in BOARD_FILES and square[1] in BOARD_RANKS

    def _place(self, sq: str, color: str, kind: str) -> None: self.board[self.parse_square(sq)] = Piece(color, kind)
    def _setup_initial_position(self) -> None:
        for sq, kind in {"a0":"R","b0":"H","c0":"E","d0":"A","e1":"K","f0":"A","g0":"E","h0":"H","i0":"R","b2":"C","h2":"C","a3":"P","c3":"P","e3":"P","g3":"P","i3":"P"}.items(): self._place(sq, "w", kind)
        for sq, kind in {"a9":"R","b9":"H","c9":"E","d9":"A","e8":"K","f9":"A","g9":"E","h9":"H","i9":"R","b7":"C","h7":"C","a6":"P","c6":"P","e6":"P","g6":"P","i6":"P"}.items(): self._place(sq, "b", kind)

    def piece_at(self, square: Tuple[int, int]) -> Optional[Piece]: return self.board.get(square)
    def side_to_move(self, ply_index: int) -> str:
        base = 0 if self._initial_color == "w" else 1
        return "w" if (ply_index + base) % 2 == 0 else "b"

    def legal_moves_for_piece(self, origin: Tuple[int, int]) -> Iterable[Tuple[int, int]]:
        piece = self.board.get(origin)
        if not piece: return []
        if piece.kind == "R": yield from self._sliding_moves(origin, piece, True, True)
        elif piece.kind == "C": yield from self._cannon_moves(origin, piece)
        elif piece.kind == "H": yield from self._horse_moves(origin, piece)
        elif piece.kind == "E": yield from self._elephant_moves(origin, piece)
        elif piece.kind in ("A", "K"): yield from self._advisor_king_moves(origin, piece)
        elif piece.kind == "P": yield from self._pawn_moves(origin, piece)

    def _sliding_moves(self, origin, piece, orth, diag):
        if orth:
            for df, dr in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                f, r = origin
                while True:
                    f += df; r += dr; sq = (f, r)
                    if not self.in_bounds(sq): break
                    blocker = self.piece_at(sq)
                    if blocker is None: yield sq; continue
                    if blocker.color != piece.color: yield sq
                    break
        if diag: yield from self._palace_diagonal_steps(origin, piece)

    def _cannon_moves(self, origin, piece):
        for df, dr in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            f, r = origin; jumped = False
            while True:
                f += df; r += dr; sq = (f, r)
                if not self.in_bounds(sq): break
                blocker = self.piece_at(sq)
                if not jumped:
                    if blocker is None: continue
                    if blocker.kind == "C": break
                    jumped = True; continue
                if blocker is None: yield sq; continue
                if blocker.color != piece.color and blocker.kind != "C": yield sq
                break
        palaces = [((4, 1), {(3, 0): (5, 2), (5, 0): (3, 2), (3, 2): (5, 0), (5, 2): (3, 0)}), ((4, 8), {(3, 7): (5, 9), (5, 7): (3, 9), (3, 9): (5, 7), (5, 9): (3, 7)})]
        for center, corners in palaces:
            if origin in corners:
                dest = corners[origin]
                c_piece = self.piece_at(center)
                if c_piece is not None and c_piece.kind != "C":
                    d_piece = self.piece_at(dest)
                    if d_piece is None or (d_piece.color != piece.color and d_piece.kind != "C"): yield dest

    def _horse_moves(self, origin, piece):
        for leg, dest_delta in [((0,1),(-1,2)), ((0,1),(1,2)), ((0,-1),(-1,-2)), ((0,-1),(1,-2)), ((1,0),(2,-1)), ((1,0),(2,1)), ((-1,0),(-2,-1)), ((-1,0),(-2,1))]:
            leg_sq = (origin[0]+leg[0], origin[1]+leg[1])
            dest = (origin[0]+dest_delta[0], origin[1]+dest_delta[1])
            if not self.in_bounds(dest) or self.piece_at(leg_sq) is not None: continue
            blocker = self.piece_at(dest)
            if blocker is None or blocker.color != piece.color: yield dest

    def _elephant_moves(self, origin, piece):
        for l1, l2_d, dest_d in [((0,1),(-1,2),(-2,3)), ((0,1),(1,2),(2,3)), ((0,-1),(-1,-2),(-2,-3)), ((0,-1),(1,-2),(2,-3)), ((1,0),(2,-1),(3,-2)), ((1,0),(2,1),(3,2)), ((-1,0),(-2,-1),(-3,-2)), ((-1,0),(-2,1),(-3,2))]:
            l1_sq = (origin[0]+l1[0], origin[1]+l1[1])
            l2_sq = (origin[0]+l2_d[0], origin[1]+l2_d[1])
            dest = (origin[0]+dest_d[0], origin[1]+dest_d[1])
            if not self.in_bounds(dest): continue
            if self.piece_at(l1_sq) is not None or self.piece_at(l2_sq) is not None: continue
            blocker = self.piece_at(dest)
            if blocker is None or blocker.color != piece.color: yield dest

    def _advisor_king_moves(self, origin, piece):
        moves = [(origin[0]+1, origin[1]), (origin[0]-1, origin[1]), (origin[0], origin[1]+1), (origin[0], origin[1]-1)]
        if origin in ((3, 0), (5, 0), (3, 2), (5, 2)): moves.append((4, 1))
        elif origin == (4, 1): moves.extend([(3, 0), (5, 0), (3, 2), (5, 2)])
        if origin in ((3, 7), (5, 7), (3, 9), (5, 9)): moves.append((4, 8))
        elif origin == (4, 8): moves.extend([(3, 7), (5, 7), (3, 9), (5, 9)])
        for dest in moves:
            if not self._in_palace(dest, piece.color): continue
            blocker = self.piece_at(dest)
            if blocker is None or blocker.color != piece.color: yield dest

    def _pawn_moves(self, origin, piece):
        dir = 1 if piece.color == "w" else -1
        moves = [(origin[0], origin[1]+dir), (origin[0]-1, origin[1]), (origin[0]+1, origin[1])]
        if piece.color == "w":
            if origin in ((3, 7), (5, 7)): moves.append((4, 8))
            elif origin == (4, 8): moves.extend([(3, 9), (5, 9)])
        else:
            if origin in ((3, 2), (5, 2)): moves.append((4, 1))
            elif origin == (4, 1): moves.extend([(3, 0), (5, 0)])
        for dest in moves:
            if not self.in_bounds(dest): continue
            blocker = self.piece_at(dest)
            if blocker is None or blocker.color != piece.color: yield dest

    def _palace_diagonal_steps(self, origin, piece):
        palaces = [((4, 1), [(3, 0), (5, 2)], [(5, 0), (3, 2)]), ((4, 8), [(3, 7), (5, 9)], [(5, 7), (3, 9)])]
        for center, diag1, diag2 in palaces:
            if origin == center:
                for dest in diag1 + diag2:
                    blocker = self.piece_at(dest)
                    if blocker is None or blocker.color != piece.color: yield dest
            else:
                for diag in (diag1, diag2):
                    if origin in diag:
                        blocker = self.piece_at(center)
                        if blocker is None or blocker.color != piece.color: yield center
                        if blocker is None:
                            opp = diag[1] if diag[0] == origin else diag[0]
                            opp_blocker = self.piece_at(opp)
                            if opp_blocker is None or opp_blocker.color != piece.color: yield opp

    def _in_palace(self, sq, color):
        return (3 <= sq[0] <= 5 and 0 <= sq[1] <= 2) if color == "w" else (3 <= sq[0] <= 5 and 7 <= sq[1] <= 9)

    def is_in_check(self, color: str) -> bool:
        king_sq = None
        for sq, p in self.board.items():
            if p.color == color and p.kind == "K":
                king_sq = sq
                break
        if not king_sq: return False

        opp_color = "b" if color == "w" else "w"
        opp_king_sq = None
        
        for sq, p in self.board.items():
            if p.color == opp_color:
                if p.kind == "K":
                    opp_king_sq = sq
                    continue
                if king_sq in self.legal_moves_for_piece(sq):
                    return True

        if opp_king_sq and king_sq[0] == opp_king_sq[0]:
            min_r = min(king_sq[1], opp_king_sq[1])
            max_r = max(king_sq[1], opp_king_sq[1])
            if not any((king_sq[0], r) in self.board for r in range(min_r + 1, max_r)):
                return True

        return False

    def move_to_uci(self, san: str, ply_index: int) -> str:
        san_clean = san.strip()
        # [NEW] 한 수 쉼(--) 예외 처리 추가
        if san_clean == "--":
            return "0000"
            
        match = MOVE_RE.match(san_clean)
        if not match: raise ValueError(f"Unsupported move token: {san}")
        
        p_kind = match.group(1).upper() if match.group(1) else "P"
        origin_hint = match.group(2).lower() 
        color = self.side_to_move(ply_index)
        dest = self.parse_square(match.group(3) + match.group(4))

        candidates = []
        for square, piece in self.board.items():
            if piece.color == color and piece.kind == p_kind and dest in self.legal_moves_for_piece(square):
                candidates.append(square)

        if origin_hint:
            filtered = []
            for sq in candidates:
                f_char = FILES[sq[0]]
                r_char = str(sq[1])
                if origin_hint == f_char or origin_hint == r_char or origin_hint == f_char + r_char:
                    filtered.append(sq)
            if filtered: candidates = filtered

        if len(candidates) > 1:
            valid_candidates = []
            for sq in candidates:
                moved_piece = self.board.pop(sq)
                captured_piece = self.board.get(dest)
                self.board[dest] = moved_piece
                
                if not self.is_in_check(color):
                    valid_candidates.append(sq)
                    
                self.board.pop(dest)
                self.board[sq] = moved_piece
                if captured_piece is not None:
                    self.board[dest] = captured_piece
                    
            if valid_candidates:
                candidates = valid_candidates

        if len(candidates) > 1:
            if p_kind == "P" and not origin_hint:
                same_file = [sq for sq in candidates if sq[0] == dest[0]]
                if same_file: candidates = same_file

        if not candidates:
            raise ValueError(f"'{san}' (해당 기물이 보드에 없거나 목적지로 이동할 수 있는 유효한 기물이 없습니다.)")
            
        if len(candidates) > 1:
            raise ValueError(f"'{san}' (이동 가능한 동일 기물이 여러 개 있어 기보가 모호합니다. 추측 이동을 방지합니다.)")
            
        origin = candidates[0]
        uci = f"{FILES[origin[0]]}{origin[1]}{FILES[dest[0]]}{dest[1]}"
        self.board[dest] = self.board.pop(origin)
        return uci

def parse_score(cp_str: str, mate_str: str) -> int:
    if cp_str: return int(cp_str)
    if mate_str:
        val = int(mate_str)
        return 10000 - val if val > 0 else -10000 - val
    return 0

class EngineSession:
    def __init__(self, engine_path: Path, depth: Optional[int], nodes: Optional[int], nnue_path: Optional[str] = None, use_nnue: bool = True, threads: int = 1) -> None:
        self.depth = depth
        self.nodes = nodes
        self.proc = subprocess.Popen([str(engine_path)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self.send("uci")
        self._wait_for("uciok")
        self.send("setoption name UCI_Variant value janggimodern")
        self.send("setoption name MultiPV value 3")
        if threads > 1: self.send(f"setoption name Threads value {threads}")
        
        if use_nnue:
            if nnue_path:
                self.send(f"setoption name EvalFile value {nnue_path}")
            self.send("setoption name Use NNUE value true")
        else:
            self.send("setoption name Use NNUE value false")
            
        self.send("isready")
        self._wait_for("readyok")

    def send(self, line: str) -> None:
        try:
            self.proc.stdin.write(line + "\n")
            self.proc.stdin.flush()
        except OSError:
            pass 

    def _wait_for(self, needle: str, timeout: float = 15.0) -> str:
        start_time = time.time()
        while True:
            if self.proc.poll() is not None:
                raise RuntimeError(f"엔진 프로세스가 비정상 종료되었습니다. (대기열: {needle})")
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("엔진과의 통신이 끊어졌습니다. (Engine Crashed)")
            if needle in line: 
                return line
            if time.time() - start_time > timeout:
                raise TimeoutError(f"엔진 응답 지연 (Timeout): '{needle}'를 찾을 수 없습니다.")

    def get_top_moves(self, fen: Optional[str], moves: Sequence[str], timeout: float = 60.0) -> Tuple[List[EngineMoveInfo], int]:
        fs_moves = [to_fs_uci(m) for m in moves]
        joined = " ".join(fs_moves)
        
        if fen:
            parts = fen.split()
            norm_fen = f"{parts[0]} {parts[1]} - - 0 1" if len(parts) >= 2 else fen
            cmd = f"position fen {norm_fen} moves {joined}" if joined else f"position fen {norm_fen}"
        else:
            cmd = f"position startpos moves {joined}" if joined else "position startpos"
            
        self.send(cmd)
        
        if self.nodes:
            self.send(f"go nodes {self.nodes}")
        else:
            self.send(f"go depth {self.depth}")

        moves_dict = {}
        best_score = 0
        current_depth = 0
        
        start_time = time.time()
        while True:
            if self.proc.poll() is not None:
                raise RuntimeError("분석 중 엔진 프로세스가 비정상 종료되었습니다.")
            line = self.proc.stdout.readline()
            if not line: 
                raise RuntimeError("분석 중 엔진과의 연결이 끊어졌습니다. (Engine Crashed)")
            if time.time() - start_time > timeout:
                raise TimeoutError(f"엔진 탐색 시간 초과 ({timeout}초). 응답이 없습니다.")
            
            m_depth = re.search(r"\bdepth (\d+)\b", line)
            if m_depth:
                d = int(m_depth.group(1))
                if d > current_depth:
                    current_depth = d
                    moves_dict.clear()

            if "info" in line and "pv" in line and "score" in line:
                score_match = re.search(r"score (cp|mate) (-?\d+)", line)
                pv_match = re.search(r"pv\s+([a-i]\d+[a-i]\d+)", line)
                if score_match and pv_match:
                    stype, sval = score_match.group(1), score_match.group(2)
                    score = parse_score(sval if stype == "cp" else None, sval if stype == "mate" else None)
                    move_uci = from_fs_uci(pv_match.group(1))
                    moves_dict[move_uci] = score
                    
            if line.startswith("bestmove"):
                sorted_moves = sorted([EngineMoveInfo(k, v) for k, v in moves_dict.items()], key=lambda x: x.score, reverse=True)
                top_3 = sorted_moves[:3]
                best_score = top_3[0].score if top_3 else 0
                return top_3, best_score

    def evaluate_specific_move(self, fen: Optional[str], moves: Sequence[str], target_move: str, timeout: float = 60.0) -> int:
        fs_moves = [to_fs_uci(m) for m in moves]
        fs_target = to_fs_uci(target_move)
        joined = " ".join(fs_moves)
        
        if fen:
            parts = fen.split()
            norm_fen = f"{parts[0]} {parts[1]} - - 0 1" if len(parts) >= 2 else fen
            cmd = f"position fen {norm_fen} moves {joined}" if joined else f"position fen {norm_fen}"
        else:
            cmd = f"position startpos moves {joined}" if joined else "position startpos"
            
        self.send(cmd)
        if self.nodes:
            self.send(f"go nodes {self.nodes} searchmoves {fs_target}")
        else:
            self.send(f"go depth {self.depth} searchmoves {fs_target}")
        
        score = 0
        start_time = time.time()
        while True:
            if self.proc.poll() is not None:
                raise RuntimeError("특정 수 평가 중 엔진 프로세스가 비정상 종료되었습니다.")
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("특정 수 평가 중 엔진과의 연결이 끊어졌습니다.")
            if time.time() - start_time > timeout:
                raise TimeoutError(f"엔진 탐색 시간 초과 ({timeout}초). 응답이 없습니다.")
                
            if "info" in line and "score" in line:
                m = re.search(r"score (cp|mate) (-?\d+)", line)
                if m: score = parse_score(m.group(2) if m.group(1) == "cp" else None, m.group(2) if m.group(1) == "mate" else None)
            if line.startswith("bestmove"):
                return score

    def close(self) -> None:
        if self.proc.poll() is None:
            try: self.send("quit")
            except: pass
            self.proc.terminate()
            self.proc.wait(timeout=5)

def tokenize_pgn_moves(text: str) -> List[str]:
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = re.sub(r"\{[^}]*\}", " ", text)
    text = re.sub(r"\([^)]*\)", " ", text)
    tokens = []
    for raw in text.replace("\n", " ").split():
        if raw in {"1-0", "0-1", "1/2-1/2", "*"}: continue
        if re.match(r"^\d+\.*$", raw): continue
        clean = raw.replace("x", "").replace("X", "").replace("+", "").replace("#", "")
        if clean: tokens.append(clean)
    return tokens

def analyze_game(engine_path: Path, pgn_text: str, depth: Optional[int], nodes: Optional[int], nnue_path: Optional[str], aux_nnues: List[str], use_nnue: bool, threads: int, log_prefix: str = "", game_id: int = 0) -> List[HistoryEntry]:
    board = JanggiBoard()
    engine = EngineSession(engine_path, depth, nodes, nnue_path, use_nnue, threads)
    
    aux_sessions = []
    if use_nnue and aux_nnues:
        print(f"{log_prefix}➔ AI 다각도 합의(Consensus) 패널 엔진 {len(aux_nnues)}개 추가 로드 중...")
        for aux_path in aux_nnues[:2]: 
            aux_sessions.append(EngineSession(engine_path, depth, nodes, aux_path, True, max(1, threads // 2)))

    fen_match = re.search(r'\[FEN\s+"([^"]+)"\]', pgn_text)
    start_fen = fen_match.group(1) if fen_match else None
    if start_fen: board.load_fen(start_fen)

    tokens = tokenize_pgn_moves(pgn_text)
    uci_moves = []
    history = []
    
    search_type = f"{nodes} Nodes" if nodes else f"Depth {depth}"
    print(f"{log_prefix}총 {len(tokens)}수 정밀 분석({search_type}, 강제수 필터링, 합의 패널 가동)을 시작합니다...")
    
    try:
        for ply, san in enumerate(tokens):
            try:
                actual_uci = board.move_to_uci(san, ply)
            except ValueError as e:
                print(f"\n\n[Warning] 기보 인식 오류 발생 ({san}) (Ply {ply+1}, {'백' if ply % 2 == 0 else '흑'} 차례)")
                print(f"상세 원인: {e}\n==> 해당 수 이후의 좌표를 명확히 인식할 수 없어 이전 수순(총 {ply}수)까지만을 기준으로 정상 분석합니다.\n")
                break
                
            try:
                top_moves, best_score = engine.get_top_moves(start_fen, uci_moves)
                top_ucis = [m.uci for m in top_moves]
                best_uci = top_ucis[0] if top_ucis else ""
                
                actual_score = None
                for m in top_moves:
                    if m.uci == actual_uci:
                        actual_score = m.score
                        break
                        
                if actual_score is None:
                    actual_score = engine.evaluate_specific_move(start_fen, uci_moves, actual_uci)
            except (RuntimeError, TimeoutError) as e:
                print(f"\n\n[Error] AI 엔진 치명적 오류 발생: {e}")
                print(f"==> 프로세스 보호를 위해 해당 수순(Ply {ply+1})까지만 분석을 기록하고 안전하게 중단합니다.\n")
                break
            
            loss = max(0, cap_score(best_score) - cap_score(actual_score))
            loss = min(loss, 300)
            
            if loss >= 15 and actual_uci != best_uci and aux_sessions:
                for aux_eng in aux_sessions:
                    aux_top, aux_best_score = aux_eng.get_top_moves(start_fen, uci_moves)
                    aux_top_ucis = [m.uci for m in aux_top]
                    
                    if actual_uci in aux_top_ucis:
                        aux_actual = next((m.score for m in aux_top if m.uci == actual_uci), None)
                        if aux_actual is None: continue
                        
                        aux_loss = max(0, cap_score(aux_best_score) - cap_score(aux_actual))
                        aux_loss = min(aux_loss, 300)
                        
                        if aux_loss < loss:
                            loss = aux_loss
                            best_uci = aux_top_ucis[0] if aux_top_ucis else best_uci
                            actual_score = aux_actual
                            for u in aux_top_ucis:
                                if u not in top_ucis: top_ucis.append(u)
                            
                            if actual_uci == best_uci:
                                break
            
            is_critical = False
            is_forced = False
            eval_gap = 9999
            
            if len(top_moves) >= 2:
                eval_gap = abs(cap_score(top_moves[0].score) - cap_score(top_moves[1].score))
                forced_threshold = 200 if ply >= 60 else 300
                if eval_gap >= forced_threshold: is_forced = True
                elif eval_gap < 150 and abs(cap_score(best_score)) < 1500: is_critical = True
            elif len(top_moves) == 1: is_forced = True

            history.append(HistoryEntry(
                ply=ply, san=san, uci=actual_uci,
                best_move=best_uci, top_3_moves=top_ucis,
                is_top_1=(actual_uci == best_uci),
                is_top_3=(actual_uci in top_ucis),
                eval_loss=loss, current_eval=actual_score,
                is_critical=is_critical, is_forced=is_forced, eval_gap=eval_gap,
                start_fen=start_fen, uci_history=list(uci_moves), game_id=game_id
            ))
            
            uci_moves.append(actual_uci)
            mark = "F" if is_forced else ("C" if is_critical else "N")
            print(f"Analyzing... [{ply+1:03d}/{len(tokens):03d}] [실제: {actual_uci} | AI: {best_uci} | Loss: {loss:3d}cp | {mark}]    ", end='\r')
            
    finally:
        print("\nAnalysis Step Complete.                        ")
        engine.close()
        for aux_eng in aux_sessions:
            aux_eng.close()
    return history

def get_numeric_phase_acpl(moves, start_ply, end_ply):
    phase_moves = [m for m in moves if start_ply <= m.ply <= end_ply]
    if not phase_moves: return None
    return sum(m.eval_loss for m in phase_moves) / len(phase_moves)

def calc_stats(moves):
    total = len(moves)
    if total == 0: return [0.0]*11
    
    losses = [m.eval_loss for m in moves]
    acpl = sum(losses) / total
    variance = sum((x - acpl) ** 2 for x in losses) / total
    std_dev = math.sqrt(variance)
    
    top1 = sum(1 for m in moves if m.is_top_1) / total * 100
    top3 = sum(1 for m in moves if m.is_top_3) / total * 100
    
    perfect = sum(1 for x in losses if x <= 5)
    inacc = sum(1 for x in losses if 50 <= x < 100)
    mistake = sum(1 for x in losses if 100 <= x < 200)
    blunder = sum(1 for x in losses if x >= 200)

    unforced = [m for m in moves if not m.is_forced]
    u_total = len(unforced)
    u_top1 = (sum(1 for m in unforced if m.is_top_1) / u_total * 100) if u_total > 0 else 0.0
    u_acpl = (sum(m.eval_loss for m in unforced) / u_total) if u_total > 0 else 0.0

    critical = [m for m in moves if m.is_critical]
    c_top1 = (sum(1 for m in critical if m.is_top_1) / len(critical) * 100) if critical else 0.0
    
    return acpl, top1, top3, perfect, inacc, mistake, blunder, std_dev, c_top1, u_top1, u_acpl

def calc_advanced_stats(moves: List[HistoryEntry]) -> Tuple[Optional[float], Optional[float], Optional[float], int]:
    unforced = [m for m in moves if not m.is_forced]
    mid_late_unforced = [m for m in unforced if m.ply >= 12]
    
    window_size = 10 
    min_window_acpl = None
    if len(mid_late_unforced) >= window_size:
        min_acpl = 999.0
        for i in range(len(mid_late_unforced) - window_size + 1):
            window = mid_late_unforced[i:i+window_size]
            w_acpl = sum(m.eval_loss for m in window) / window_size
            if w_acpl < min_acpl:
                min_acpl = w_acpl
        min_window_acpl = min_acpl

    hard_moves = [m for m in unforced if m.eval_gap <= 50 and abs(m.current_eval) < 1500]
    hard_count = len(hard_moves)
    hard_top1 = (sum(1 for m in hard_moves if m.is_top_1) / hard_count * 100) if hard_count > 0 else None

    recovery_score = 0
    blunder_count = 0
    for i, m in enumerate(moves):
        if m.eval_loss >= 200:
            next_moves = moves[i+1:i+4]
            if len(next_moves) == 3:
                blunder_count += 1
                recovery_acpl = sum(nx.eval_loss for nx in next_moves) / 3
                if recovery_acpl <= 10:
                    recovery_score += 1
    
    recovery_rate = (recovery_score / blunder_count * 100) if blunder_count > 0 else None

    return min_window_acpl, hard_top1, recovery_rate, hard_count

def _rank_weighted_match(move: HistoryEntry) -> float:
    """Janggi-safe rank score: Top-1 matters most; raw Top-3 is intentionally discounted."""
    if move.uci == move.best_move:
        return 1.0
    if move.uci in move.top_3_moves:
        rank = move.top_3_moves.index(move.uci)
        if rank == 1:
            return 0.45
        if rank == 2:
            return 0.20
    return 0.0

def calculate_tactical_burst(moves: List[HistoryEntry]) -> Tuple[float, List[TacticalBurstResult], str]:
    """
    Conservative detector for short Janggi tactical/rescue bursts.
    It only inspects same-game, non-forced, hard/critical moves and uses rank-weighted
    matching so naturally inflated Janggi Top-3 rates do not dominate the score.
    """
    if len(moves) < 6:
        return 0.0, [], "데이터 부족"

    def clamp_map(val, in_min, in_max, out_min, out_max):
        if in_min < in_max:
            if val <= in_min: return out_min
            if val >= in_max: return out_max
            return out_min + (val - in_min) * (out_max - out_min) / (in_max - in_min)
        else:
            if val >= in_min: return out_min
            if val <= in_max: return out_max
            return out_min + (in_min - val) * (out_max - out_min) / (in_min - in_max)

    by_game: Dict[int, List[HistoryEntry]] = {}
    for m in sorted(moves, key=lambda x: (x.game_id, x.ply)):
        by_game.setdefault(m.game_id, []).append(m)

    scored: List[TacticalBurstResult] = []

    for game_id, game_moves in by_game.items():
        unforced = [m for m in game_moves if not m.is_forced]
        if len(unforced) < 6:
            continue

        baseline_losses = [m.eval_loss for m in unforced]
        baseline_acpl_all = sum(baseline_losses) / len(baseline_losses)
        baseline_rank_all = sum(_rank_weighted_match(m) for m in unforced) / len(unforced)
        baseline_std_all = math.sqrt(sum((x - baseline_acpl_all) ** 2 for x in baseline_losses) / len(baseline_losses))

        difficult = []
        for m in game_moves:
            if m.is_forced or m.ply < 12 or abs(m.current_eval) >= 1800:
                continue
            # Avoid overvaluing naturally forcing Janggi continuations: a perfect move is
            # not treated as difficult unless the engine gap also indicates real choice.
            is_hard = m.eval_gap <= 60
            is_tactical_close = m.is_critical and m.eval_gap <= 90
            has_candidate_choice = len(m.top_3_moves) >= 2
            if not has_candidate_choice or not (is_hard or is_tactical_close):
                continue
            difficult.append(m)

        for size in range(3, 7):
            if len(difficult) < size:
                continue
            for start in range(len(difficult) - size + 1):
                window = difficult[start:start + size]
                if window[-1].ply - window[0].ply > 18:
                    continue

                losses = [m.eval_loss for m in window]
                w_acpl = sum(losses) / size
                w_rank = sum(_rank_weighted_match(m) for m in window) / size
                w_top1 = sum(1 for m in window if m.is_top_1) / size * 100
                w_std = math.sqrt(sum((x - w_acpl) ** 2 for x in losses) / size)
                hard_count = sum(1 for m in window if m.eval_gap <= 50 or (m.is_critical and m.eval_gap <= 90))
                required_hard = size if size <= 4 else size - 1
                if hard_count < required_hard:
                    continue

                outside = [m for m in unforced if not (window[0].ply <= m.ply <= window[-1].ply)]
                if len(outside) >= 4:
                    out_losses = [m.eval_loss for m in outside]
                    baseline_acpl = sum(out_losses) / len(out_losses)
                    baseline_rank = sum(_rank_weighted_match(m) for m in outside) / len(outside)
                    baseline_std = math.sqrt(sum((x - baseline_acpl) ** 2 for x in out_losses) / len(out_losses))
                else:
                    baseline_acpl = baseline_acpl_all
                    baseline_rank = baseline_rank_all
                    baseline_std = baseline_std_all

                swing_count = 0
                recovery_count = 0
                for m in window:
                    prev_candidates = [pm for pm in game_moves if pm.ply < m.ply]
                    prev = prev_candidates[-1] if prev_candidates else None
                    if prev:
                        if abs(m.current_eval - prev.current_eval) >= 300:
                            swing_count += 1
                        if prev.eval_loss >= 200:
                            recovery_count += 1

                similar_count = 0
                for j in range(len(difficult) - size + 1):
                    other = difficult[j:j + size]
                    if other[-1].ply - other[0].ply > 18:
                        continue
                    other_acpl = sum(x.eval_loss for x in other) / size
                    other_rank = sum(_rank_weighted_match(x) for x in other) / size
                    if other_acpl <= 8.0 and other_rank >= 0.70:
                        similar_count += 1

                relative_gain = baseline_acpl - w_acpl
                rank_gain = w_rank - baseline_rank
                variance_gain = baseline_std - w_std
                has_context = bool(swing_count or recovery_count)

                # Perfect short windows are common in professional Janggi.  The score is
                # therefore mostly relative to the player's surrounding level, not absolute ACPL 0.
                acpl_component = clamp_map(w_acpl, 6.0, 0.0, 0.0, 8.0)
                improvement_component = clamp_map(relative_gain, 18.0, 60.0, 0.0, 34.0)
                rank_component = clamp_map(rank_gain, 0.18, 0.50, 0.0, 18.0)
                consistency_component = clamp_map(variance_gain, 12.0, 45.0, 0.0, 12.0)
                context_component = min(12.0, swing_count * 5.0 + recovery_count * 6.0)
                rarity_component = 8.0 if similar_count == 1 and relative_gain >= 22.0 else (3.0 if similar_count == 2 and relative_gain >= 30.0 else 0.0)

                raw_score = acpl_component + improvement_component + rank_component + consistency_component + context_component + rarity_component

                length_weight = {3: 0.30, 4: 0.50, 5: 0.78, 6: 1.00}[size]
                raw_score *= length_weight

                if baseline_acpl <= 18.0 and 12.0 <= baseline_std <= 45.0 and baseline_rank >= 0.55:
                    raw_score *= 0.35 if size <= 4 else 0.55
                elif baseline_acpl <= 24.0 and baseline_rank >= 0.60:
                    raw_score *= 0.65 if size <= 4 else 0.80
                if baseline_rank >= 0.72:
                    raw_score *= 0.70
                if similar_count >= 3:
                    raw_score *= 0.65
                if relative_gain < 15.0:
                    raw_score *= 0.40
                if rank_gain < 0.12 and not has_context:
                    raw_score *= 0.55
                if w_acpl > 6.0 or w_rank < 0.72:
                    raw_score *= 0.50
                if w_std > 10.0:
                    raw_score *= 0.70

                cap = {3: 24.0, 4: 42.0, 5: 68.0, 6: 82.0}[size]
                if has_context and relative_gain >= 25.0:
                    cap += {3: 8.0, 4: 10.0, 5: 8.0, 6: 6.0}[size]
                if relative_gain < 25.0 or rank_gain < 0.18:
                    cap = min(cap, 40.0 if size <= 4 else 58.0)

                score = min(cap, max(0.0, raw_score))
                if score < 35.0:
                    continue

                reasons = []
                if w_acpl <= 4.0: reasons.append(f"단기 ACPL {w_acpl:.1f}")
                if rank_gain >= 0.18: reasons.append(f"주변 대비 순위가중 +{rank_gain*100:.0f}%")
                if relative_gain >= 18.0: reasons.append(f"주변 대비 손실 {relative_gain:.1f} 감소")
                if variance_gain >= 12.0: reasons.append(f"단기 편차 {w_std:.1f}로 수렴")
                if similar_count <= 2: reasons.append("게임 내 드문 정밀 구간")
                if swing_count: reasons.append(f"평가 급변 후 정밀수 {swing_count}회")
                if recovery_count: reasons.append(f"실수 직후 복구 {recovery_count}회")
                if not reasons: reasons.append("검토용 국지 정밀도 상승")

                scored.append(TacticalBurstResult(
                    score=score,
                    start_ply=window[0].ply + 1,
                    end_ply=window[-1].ply + 1,
                    game_id=game_id,
                    move_count=size,
                    acpl=w_acpl,
                    rank_match=w_rank * 100,
                    top1_rate=w_top1,
                    std_dev=w_std,
                    baseline_acpl=baseline_acpl,
                    baseline_rank=baseline_rank * 100,
                    hard_count=hard_count,
                    swing_count=swing_count,
                    recovery_count=recovery_count,
                    commentary=", ".join(reasons)
                ))

    scored.sort(key=lambda b: b.score, reverse=True)
    filtered: List[TacticalBurstResult] = []
    for burst in scored:
        overlaps = any(
            burst.game_id == kept.game_id and not (burst.end_ply < kept.start_ply or burst.start_ply > kept.end_ply)
            for kept in filtered
        )
        if not overlaps:
            filtered.append(burst)
        if len(filtered) >= 3:
            break

    if not filtered:
        return 0.0, [], "단기 전술 버스트 특이점 없음"

    best = filtered[0]
    if best.score >= 70.0:
        verdict = "다른 지표와 함께 검토할 국지 정밀도 상승"
    elif best.score >= 50.0:
        verdict = "참고용 국지 정밀도 신호"
    else:
        verdict = "약한 단기 정밀도 신호(참고용)"
    return best.score, filtered, verdict

def calculate_partial_ai_probability(moves: List[HistoryEntry]) -> Tuple[float, str, List[int]]:
    valid_unforced = [m for m in moves if not m.is_forced and 20 <= m.ply <= 90]
    
    cleaned_unforced = []
    for m in valid_unforced:
        best_eval = cap_score(m.current_eval + m.eval_loss)
        if best_eval >= 200 and m.eval_loss >= 80 and m.current_eval >= -150:
            continue 
        cleaned_unforced.append(m)
        
    window_size = 16  # 프로들의 강제/필연 수순을 고려해 검사 구간을 16수(8턴)로 늘림

    def clamp_map(val, in_min, in_max, out_min, out_max):
        if in_min < in_max:
            if val <= in_min: return out_min
            if val >= in_max: return out_max
            return out_min + (val - in_min) * (out_max - out_min) / (in_max - in_min)
        else:
            if val >= in_min: return out_min
            if val <= in_max: return out_max
            return out_min + (in_min - val) * (out_max - out_min) / (in_min - in_max)

    max_spike_score = 0.0
    suspect_range = [0, 0]
    checked_windows = 0

    by_game: Dict[int, List[HistoryEntry]] = {}
    for m in cleaned_unforced:
        by_game.setdefault(m.game_id, []).append(m)

    for game_moves in by_game.values():
        if len(game_moves) < window_size:
            continue
        for i in range(len(game_moves) - window_size + 1):
            window = game_moves[i:i + window_size]
            checked_windows += 1
            w_acpl = sum(m.eval_loss for m in window) / window_size
            w_top1 = (sum(1 for m in window if m.is_top_1) / window_size) * 100

            mistakes = sum(1 for m in window if m.eval_loss >= 100)
            inaccs = sum(1 for m in window if m.eval_loss >= 50)

            # 프로 컷 상향: ACPL이 7점 이하, Top-1이 65% 이상일 때만 점수가 오르기 시작함
            acpl_score = clamp_map(w_acpl, 7.0, 1.5, 0.0, 60.0)
            top1_score = clamp_map(w_top1, 65.0, 90.0, 0.0, 40.0)
            err_penalty = (mistakes * 20.0) + (inaccs * 10.0)

            current_spike = max(0.0, acpl_score + top1_score - err_penalty)
            if current_spike > max_spike_score:
                max_spike_score = current_spike
                suspect_range = [window[0].ply + 1, window[-1].ply + 1]

    if checked_windows == 0:
        return 0.0, "데이터 부족 (탐색 구간 짧음)", [0, 0]

    # [수정된 프로 기사 보호막 (Pro-Dampener)]
    # 프로 수준의 편차(15~35)를 가지면 스파이크 점수를 대폭 깎아줍니다.
    stats = calc_stats(moves)
    std_dev = stats[7]
    pro_dampener = 1.0
    if 15.0 <= std_dev <= 45.0:
        pro_dampener = clamp_map(std_dev, 15.0, 30.0, 0.8, 0.2) # 기복이 20대면 점수를 50% 이상 깎음

    final_spike_prob = min(99.0, max_spike_score * pro_dampener)
    
    p_verdict = "🟢 특이 구간 없음 (자연스러운 수순)"
    if final_spike_prob >= 85: p_verdict = "🔴 특정 난전 구간에서 기계적 수순 전개와 매우 흡사함"
    elif final_spike_prob >= 70: p_verdict = "🟡 특정 승부처에서 AI 추천수와의 연관성이 관찰됨"
    
    return final_spike_prob, p_verdict, suspect_range

def calculate_cheat_probability(moves) -> Tuple[float, float, str]:
    if len(moves) < 5:
        return 0.0, 0.0, "⚪ 판독 불가 (수순 부족)"

    def clamp_map(val, in_min, in_max, out_min, out_max):
        if in_min < in_max:
            if val <= in_min: return out_min
            if val >= in_max: return out_max
            return out_min + (val - in_min) * (out_max - out_min) / (in_max - in_min)
        else:
            if val >= in_min: return out_min
            if val <= in_max: return out_max
            return out_min + (in_min - val) * (out_max - out_min) / (in_min - in_max)

    avoided_moves = []
    normal_moves = []
    avoidance_weight_total = 0.0
    real_mistakes = 0
    real_blunders = 0

    for m in moves:
        best_eval = cap_score(m.current_eval + m.eval_loss)
        
        # [핵심 개선] 진짜 '의도적 회피(스마트 치팅)'의 조건 필터링
        is_intentional_avoidance = (
            best_eval >= 400 and           # 1. 원래 두면 확정적으로 크게 유리한 상황 (단순 2점이 아닌 확실한 승기)
            m.eval_loss >= 100 and         # 2. 그 1위 수를 안 둬서 명백히 점수 손해를 봄
            m.current_eval >= 150 and      # 3. 그런데 대신 둔 수조차도 여전히 형세가 좋음 (완전한 실수가 아님)
            m.is_top_3 and not m.is_top_1  # 4. [가장 중요] 사람만의 수를 둔 게 아니라, 기계적인 차선책(AI 2~3위수)을 둔 경우
        )

        if is_intentional_avoidance:
            avoided_moves.append(m)
            weight = 1.0
            if m.eval_loss >= 300: weight += 1.0 
            if best_eval >= 800: weight += 1.0   
            avoidance_weight_total += weight
        else:
            normal_moves.append(m)
            # 사람이 좋은 수를 진짜로 못 보고 놓친 경우, 아래에서 실수(Mistake/Blunder)로 카운트되어
            # 사람 지수(humanity_index)가 올라가고 치팅 확률이 오히려 내려가게 됩니다.
            if m.eval_loss >= 200: real_blunders += 1
            elif m.eval_loss >= 100: real_mistakes += 1

    if not normal_moves: normal_moves = moves
    
    adj_losses = [m.eval_loss for m in normal_moves]
    adj_acpl = sum(adj_losses) / len(adj_losses) if adj_losses else 0.0
    adj_unforced = [m for m in normal_moves if not m.is_forced]
    adj_u_top1 = (sum(1 for m in adj_unforced if m.is_top_1) / len(adj_unforced) * 100) if adj_unforced else 0.0
    adj_top3 = (sum(1 for m in normal_moves if m.is_top_3) / len(normal_moves) * 100) if normal_moves else 0.0

    em_moves = [m for m in normal_moves if m.ply <= 69]
    if not em_moves: em_moves = normal_moves
    em_acpl = sum(m.eval_loss for m in em_moves) / len(em_moves) if em_moves else adj_acpl

    # 초고수 프로(ACPL 9~12, Top-1 50~60%)가 0%~10%대 정상 수치를 받도록 기준 상향
    acpl_score = clamp_map(em_acpl, 9.0, 3.0, 0.0, 100.0)
    match_score = max(clamp_map(adj_u_top1, 65.0, 85.0, 0.0, 100.0), clamp_map(adj_top3, 85.0, 95.0, 0.0, 100.0))
    base_prob = (acpl_score * 0.55) + (match_score * 0.45)

    bot_armor = clamp_map(em_acpl, 10.0, 3.0, 0.0, 1.0)
    if real_blunders >= 1: bot_armor *= 0.2

    stats = calc_stats(moves)
    std_dev = stats[7]
    human_err = clamp_map(real_blunders + real_mistakes, 0, 3, 0.0, 100.0)
    # 인간성 점수(편차)를 프로 기준(15~35)에 맞춰 넉넉하게 보장
    human_var = clamp_map(std_dev, 15.0, 35.0, 0.0, 100.0) 
    humanity_index = (human_err + human_var) / 2.0
    
    full_prob = base_prob - (humanity_index * 0.5 * (1.0 - bot_armor)) # 차감율 상향

    # -------------------------------------------------------------------------
    # 4. 스마트 치팅 정밀 판독 (프로 보호막 Variance Gate 적용)
    # -------------------------------------------------------------------------
    centaur_prob = 0.0
    centaur_flags = []

    variance_gate = clamp_map(std_dev, 45.0, 70.0, 0.1, 1.0)
    if avoidance_weight_total >= 7.5:
        variance_gate = max(0.8, variance_gate)

    # A. 의도적 이득 회피 (안전수)
    if avoidance_weight_total >= 3.0:
        if adj_acpl <= 35.0: # 프로의 일반적 손실(35) 이하일 때만 작동
            s_avoid = clamp_map(avoidance_weight_total, 3.0, 10.0, 15.0, 60.0)
            s_acpl = clamp_map(adj_acpl, 35.0, 10.0, 10.0, 40.0)
            score = (s_avoid + s_acpl) * variance_gate
            centaur_prob += score
            if score >= 20.0:
                centaur_flags.append(f"이득 회피/기계적 차선책(가중치 {avoidance_weight_total:.1f})")

    # B. 페이즈 격차 (초반 하수 -> 중반 고수)
    early_moves = [m for m in normal_moves if m.ply <= 29]
    mid_moves = [m for m in normal_moves if 30 <= m.ply <= 69]
    e_acpl = sum(m.eval_loss for m in early_moves)/len(early_moves) if early_moves else adj_acpl
    m_acpl = sum(m.eval_loss for m in mid_moves)/len(mid_moves) if mid_moves else adj_acpl
    
    phase_diff = e_acpl - m_acpl
    if phase_diff >= 35.0 and m_acpl <= 20.0:
        variance_gate = max(0.9, variance_gate) 

    if phase_diff >= 22.0 and m_acpl <= 20.0: # 중반 ACPL이 20 이하(기계적)일 때만 작동
        added = clamp_map(phase_diff, 22.0, 55.0, 20.0, 75.0)
        score = added * variance_gate
        centaur_prob += score
        if score >= 20.0:
            centaur_flags.append(f"승부처 급격한 성능 향상(ACPL {m_acpl:.1f})")

    # C. 기계적 정답률 괴리 (수학 오류 완전 수정)
    # 사람이라면 보통 ACPL이 낮으면(잘 두면) Top-1도 같이 높아야 함. 
    # 그런데 Top-1은 비정상적으로 높고(70% 이상), ACPL도 극단적으로 낮으면(8 이하) 치팅 가중치 부여
    if adj_u_top1 >= 70.0 and adj_acpl <= 8.0:
        added = clamp_map(adj_u_top1, 70.0, 90.0, 15.0, 40.0)
        score = added * variance_gate
        centaur_prob += score
        if score >= 15.0 and not any("안전수" in f for f in centaur_flags):
            centaur_flags.append("인간을 초월한 비강제 Top-1 일치율")

    final_full_prob = max(full_prob, centaur_prob)
    final_full_prob = min(99.0, max(0.0, final_full_prob))

    partial_prob, p_verdict, s_range = calculate_partial_ai_probability(moves)
    segment_partial_prob = partial_prob
    tactical_prob, tactical_bursts, tactical_verdict = calculate_tactical_burst(moves)
    global_support = max(final_full_prob, segment_partial_prob)
    if tactical_prob >= 70.0 and global_support >= 50.0:
        # Tactical bursts are supporting evidence only; they cannot create a high
        # partial verdict when the rest of the game is statistically ordinary.
        partial_prob = max(partial_prob, min(75.0, global_support + min(10.0, (tactical_prob - 70.0) / 2.0)))

    final_verdict = ""
    if centaur_prob >= 65.0 and centaur_prob >= full_prob:
        flag_str = ", ".join(centaur_flags[:2])
        if not flag_str: flag_str = "통계적 특이 지표 다수"
        spike_info = f" [의심 구간: {s_range[0]}~{s_range[1]}수]" if segment_partial_prob >= 65.0 else ""
        final_verdict = f"🟡 [스마트 치팅 감지] 안전수 위장 속 부분적 AI 개입 발견 ({flag_str}){spike_info}"
    elif partial_prob > final_full_prob + 10.0 and partial_prob >= 65.0:
        final_verdict = f"{p_verdict} [의심 구간: {s_range[0]}~{s_range[1]}수]"
    else:
        if final_full_prob >= 85: final_verdict = "🔴 기계적 수순 패턴과 매우 높은 통계적 연관성을 보임"
        elif final_full_prob >= 75: final_verdict = "🟡 인간 최고수 범주를 상회하는 통계적 일치도가 관찰됨"
        elif final_full_prob >= 60: final_verdict = "🟠 최상위권 프로 및 고수 수준의 정교한 대국 내용"
        else: final_verdict = "🟢 보편적인 인간 대국자의 흐름 및 편차"

    return final_full_prob, partial_prob, final_verdict

def pad_korean(s: str, width: int, align: str = "center") -> str:
    visual_width = sum(2 if ord(c) > 0x7F else 1 for c in s)
    padding = max(0, width - visual_width)
    if align == "left": return s + (" " * padding)
    elif align == "right": return (" " * padding) + s
    left = padding // 2
    return (" " * left) + s + (" " * (padding - left))

def format_adv_metric(val: Optional[float], is_percent: bool = False) -> str:
    if val is None: return "-"
    return f"{val:.1f}%" if is_percent else f"{val:.1f}"

def format_tactical_burst_cell(score: float, bursts: List[TacticalBurstResult]) -> str:
    if not bursts:
        return f"{score:.1f}% / -"
    best = bursts[0]
    return f"{score:.1f}% / {best.start_ply}~{best.end_ply}수"

def print_tactical_burst_details(rows: List[Tuple[str, float, List[TacticalBurstResult], str]]) -> None:
    print("\n [ 단기 전술 버스트 참고 지표 (3~6수, 비강제·난전 한정) ]")
    header = f" {pad_korean('진영', 12)} | {pad_korean('참고 점수/구간', 22)} | {pad_korean('검토 근거', 58)}"
    print(header)
    print("-" * 105)
    for name, score, bursts, verdict in rows:
        if bursts:
            best = bursts[0]
            detail = f"{verdict} ({best.move_count}수, ACPL {best.acpl:.1f}, 순위가중 {best.rank_match:.0f}%, 주변ACPL {best.baseline_acpl:.1f})"
        else:
            detail = verdict
        print(f" {pad_korean(name, 12)} | {pad_korean(format_tactical_burst_cell(score, bursts), 22)} | {detail}")
        for extra in bursts[:2]:
            print(f" {'':12} | {'':22} | ↳ G{extra.game_id} {extra.start_ply}~{extra.end_ply}수: {extra.commentary}")

def run_files_analysis(inputs, engine_path, depth, nodes, nnue_path, aux_nnues, use_nnue, threads, is_single_no_target, log_prefix=""):
    cho_moves_all = []
    han_moves_all = []
    target_moves_all = []
    opp_moves_all = []
    white_name, black_name = "", ""

    for i, item in enumerate(inputs, 1):
        if ":" in item:
            file_path, side = item.rsplit(":", 1)
            side = side.lower()
        else:
            file_path, side = item, "both" if is_single_no_target else "cho"

        input_text = Path(file_path).read_text(encoding="utf-8")
        if i == 1: white_name, black_name = extract_player_names(input_text)
        
        game_history = analyze_game(engine_path, input_text, depth, nodes, nnue_path, aux_nnues, use_nnue, threads, log_prefix, game_id=i)
        
        if is_single_no_target:
            cho_moves_all.extend([h for h in game_history if h.ply % 2 == 0])
            han_moves_all.extend([h for h in game_history if h.ply % 2 != 0])
        else:
            if side == "cho":
                target_moves_all.extend([h for h in game_history if h.ply % 2 == 0])
                opp_moves_all.extend([h for h in game_history if h.ply % 2 != 0])
            else:
                target_moves_all.extend([h for h in game_history if h.ply % 2 != 0])
                opp_moves_all.extend([h for h in game_history if h.ply % 2 == 0])

    return cho_moves_all, han_moves_all, target_moves_all, opp_moves_all, white_name, black_name

def evaluate_single_nnue(nnue_path: Path, engine_path: Path, depth: int, nodes: Optional[int], test_targets: List[HistoryEntry]) -> Tuple[str, List[Tuple[str, List[str]]]]:
    engine = EngineSession(engine_path, depth, nodes, str(nnue_path), True, 1)
    preds = []
    for tgt in test_targets:
        top_moves, _ = engine.get_top_moves(tgt.start_fen, tgt.uci_history)
        top_ucis = [m.uci for m in top_moves]
        t1 = top_ucis[0] if top_ucis else ""
        preds.append((t1, top_ucis))
    engine.close()
    return (nnue_path.name, preds)

def run_fingerprinting(engine_path: Path, nnue_dir: str, depth: int, nodes: Optional[int], threads: int, target_moves: List[HistoryEntry], target_name: str):
    nnue_files = list(Path(nnue_dir).glob("*.nnue"))
    if not nnue_files:
        print(f"\n[오류] 지정한 폴더({nnue_dir})에서 NNUE 파일을 찾을 수 없습니다.")
        return

    candidates = [m for m in target_moves if not m.is_forced and m.eval_loss <= 5 and m.ply >= 12]
    candidates.sort(key=lambda m: m.eval_gap)
    
    target_limit = min(100, max(30, len(candidates) // 3))
    test_targets = candidates[:target_limit] 

    if not test_targets:
        print(f"\n[안내] {target_name}의 기보에는 유사 모델을 특정할 만한 '복잡한 비강제 완벽수'가 충분하지 않습니다.")
        return

    print("\n" + "="*105)
    print(pad_korean(f"★ [ 심층 분석: NNUE 핑거프린팅 (유사 모델 추적) ] - 대상: {target_name} ★", 105))
    print("="*105)
    print(f" 발견된 교차 검증용 NNUE 파일: {len(nnue_files)}개")
    print(f" 1차 추출된 변별력 높은 핵심 수순(Hard-Perfect Moves): {len(test_targets)}개")
    
    engine_results = {}
    max_workers = min(len(nnue_files), max(1, threads // 2))
    print(" 다중 스레드를 통한 고속 핑거프린팅 분석 진행 중... (기다려 주십시오)", end='\r')
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(evaluate_single_nnue, nnue, engine_path, depth, nodes, test_targets) for nnue in nnue_files]
        for idx, future in enumerate(as_completed(futures), 1):
            name, preds = future.result()
            engine_results[name] = preds
            print(f" 병렬 분석 진행 중... [{idx}/{len(nnue_files)}] 완료              ", end='\r')

    valid_divergent_indices = []
    for i in range(len(test_targets)):
        t1_set = set()
        for nnue_name in engine_results:
            t1_set.add(engine_results[nnue_name][i][0])
        
        if len(t1_set) > 1:
            valid_divergent_indices.append(i)

    divergent_count = len(valid_divergent_indices)
    is_divergent_mode = True

    if divergent_count == 0:
        valid_divergent_indices = list(range(len(test_targets)))
        divergent_count = len(valid_divergent_indices)
        is_divergent_mode = False

    if is_divergent_mode:
        print(f"\n ➔ 모델 간 편차가 존재하는(변별력 높은) 특이 수순: {divergent_count}개 필터링 완료")
    else:
        print("\n")
    print(" 유사 모델 추적 완료!                                       \n")

    results = {}
    max_possible_score = divergent_count * 3
    
    for nnue_name, preds in engine_results.items():
        t1_match = 0
        t3_match = 0
        weighted_score = 0
        
        for i in valid_divergent_indices:
            tgt_uci = test_targets[i].uci
            t1, t3_list = preds[i]
            
            if tgt_uci in t3_list:
                t3_match += 1
                rank = t3_list.index(tgt_uci)
                
                if rank == 0:
                    t1_match += 1
                    weighted_score += 3
                elif rank == 1:
                    weighted_score += 2
                elif rank == 2:
                    weighted_score += 1
                
        t1_rate = (t1_match / divergent_count) * 100 if divergent_count > 0 else 0.0
        t3_rate = (t3_match / divergent_count) * 100 if divergent_count > 0 else 0.0
        match_index = (weighted_score / max_possible_score) * 100 if max_possible_score > 0 else 0.0
        
        results[nnue_name] = (match_index, t1_rate, t3_rate)
    
    sorted_res = sorted(results.items(), key=lambda x: (x[1][0], x[1][1]), reverse=True)
    
    print(f" {pad_korean('NNUE 파일명', 35)} | {pad_korean('유사도 지수(종합)', 20)} | {pad_korean('Top-1 일치율', 15)} | {pad_korean('Top-3 포함률', 15)}")
    print("-" * 105)
    for nnue_name, (match_idx, t1_rate, t3_rate) in sorted_res:
        row = f" {pad_korean(nnue_name, 35)} | {pad_korean(f'{match_idx:.1f} 점', 20)} | {pad_korean(f'{t1_rate:.1f}%', 15)} | {pad_korean(f'{t3_rate:.1f}%', 15)}"
        print(row)
    print("="*105)

    f_prob, p_prob, _ = calculate_cheat_probability(target_moves)
    suspicion_level = max(f_prob, p_prob)
    
    top_match_idx = sorted_res[0][1][0] if sorted_res else 0
    top_nnue = sorted_res[0][0] if sorted_res else ""

    print(pad_korean("[ 핑거프린팅 종합 소견 참고사항 ]", 105))
    print("-" * 105)
    if suspicion_level < 60:
        print(" 🟢 대상의 전반적인 대국 지표가 일반적인 인간의 범주에 속합니다.")
        print("    (위 일치율은 특정 모델을 참고했다기보다는 대국자의 우수한 수읽기 결과일 가능성이 높습니다.)")
    else:
        if top_match_idx >= 80.0:
            print(f" 🔴 대상 기보에서 AI 특유의 수순 전개와 짙은 유사성이 나타나며,")
            print(f"    검토된 파일 중 [{top_nnue}] 모델(또는 동종 계열)과 가장 높은 상관관계를 보입니다.")
            print(f"    (대국 시 해당 모델의 추천수를 참고했을 가능성이 통계적으로 유의미합니다.)")
        else:
            print(" 🟠 대상 기보에서 AI와의 유사 패턴이 일부 관찰되나,")
            print("    제공된 NNUE 목록 중 특정 모델을 명확히 지목할 만큼 두드러지는 특이점은 나타나지 않았습니다.")
    print("="*105)

    print("\n[ 핑거프린팅 결과 해석 가이드 ]")
    print(" ■ 유사도 지수(종합): 단순 1위 적중률의 맹점을 보완하기 위해 1위(3점), 2위(2점), 3위(1점)의 가중치를")
    print("   부여하여 100점 만점으로 환산한 점수입니다. Top-1 일치율보다 안정적이고 신뢰도가 높습니다.")
    print(" ■ 탐색 깊이나 멀티스레드 환경에 따라 엔진의 1위 수가 미세하게 변동될 수 있으므로,")
    print("   단순히 1위 일치율(Top-1)만 보기보다는 상위 후보군(Top-3) 포함률을 함께 고려해야 합니다.")
    print(" ■ 단일 파일의 순위보다는 '유사도 지수'가 상위권에 몰려있는 모델 그룹 전체를")
    print("   대국 당시 가장 밀접하게 참고되었을 가능성이 있는 엔진 계열로 해석하는 것이 권장됩니다.")
    print("="*105)

def print_target_report(target_moves: List[HistoryEntry], opp_moves: List[HistoryEntry], game_count: int, stage_name: str, use_db: bool = False, search_config: str = "") -> None:
    if not target_moves: return
    t_stats = calc_stats(target_moves)
    o_stats = calc_stats(opp_moves)
    
    t_adv = calc_advanced_stats(target_moves)
    o_adv = calc_advanced_stats(opp_moves)

    t_fprob, t_pprob, t_verdict = calculate_cheat_probability(target_moves)
    o_fprob, o_pprob, o_verdict = calculate_cheat_probability(opp_moves)
    t_burst_score, t_bursts, t_burst_verdict = calculate_tactical_burst(target_moves)
    o_burst_score, o_bursts, o_burst_verdict = calculate_tactical_burst(opp_moves)

    def format_phase(moves, start, end):
        val = get_numeric_phase_acpl(moves, start, end)
        return f"{val:.1f}" if val is not None else "-"

    print("\n" + "="*105)
    print(pad_korean(f"장기 기보 AI 유사도 프로파일링 리포트 [ 다중 기보 누적/타겟 분석 ] - {stage_name}", 105))
    print(pad_korean(f"총 분석 게임 수: {game_count} 게임 / 타겟 분석 수순: {len(target_moves)}수", 105))
    print("="*105)
    
    print(f" [ 핵심 지표 (강제수 필터링 적용) ]")
    header = f" {pad_korean('진영', 12)} | {pad_korean('전체ACPL', 11)} | {pad_korean('비강제ACPL', 12)} | {pad_korean('편차(기복)', 11)} | {pad_korean('전체Top-3', 11)} | {pad_korean('비강제Top-1', 12)} | {pad_korean('난전Top-1', 11)}"
    print(header)
    print("-" * 105)
    for name, s in [("타겟(분석)", t_stats), ("상대방(평균)", o_stats)]:
        row = f" {pad_korean(name, 12)} | {s[0]:11.2f} | {s[10]:12.2f} | {s[7]:11.2f} | {pad_korean(f'{s[2]:.1f}%', 11)} | {pad_korean(f'{s[9]:.1f}%', 12)} | {pad_korean(f'{s[8]:.1f}%', 11)}"
        print(row)
    
    print("\n [ 세분화된 수순 평가 ]")
    header2 = f" {pad_korean('진영', 12)} | {pad_korean('완벽수(≤5cp)', 14)} | {pad_korean('의문수(50~99)', 14)} | {pad_korean('일반실수(100~199)', 14)} | {pad_korean('치명적실수(≥200)', 14)}"
    print(header2)
    print("-" * 105)
    for name, s in [("타겟(분석)", t_stats), ("상대방(평균)", o_stats)]:
        row = f" {pad_korean(name, 12)} | {pad_korean(f'{s[3]} 회', 14)} | {pad_korean(f'{s[4]} 회', 14)} | {pad_korean(f'{s[5]} 회', 14)} | {pad_korean(f'{s[6]} 회', 14)}"
        print(row)

    print("\n [ 구간별 점수 손실 (Phase Analysis) ] *(후반부 AI 평가치 변동 보정용)*")
    header3 = f" {pad_korean('진영', 12)} | {pad_korean('초반 (1~30수)', 25)} | {pad_korean('중반 (31~70수)', 25)} | {pad_korean('종반 (71수 이후)', 25)}"
    print(header3)
    print("-" * 105)
    for name, m in [("타겟(분석)", target_moves), ("상대방(평균)", opp_moves)]:
        row = f" {pad_korean(name, 12)} | {pad_korean(format_phase(m, 0, 29), 25)} | {pad_korean(format_phase(m, 30, 69), 25)} | {pad_korean(format_phase(m, 70, 999), 25)}"
        print(row)

    print("\n [ 특정 구간 AI 밀착도 심층 지표 ]")
    header4 = f" {pad_korean('진영', 12)} | {pad_korean('최고성능구간(10수) ACPL', 26)} | {pad_korean('어려운 포지션(격차<50) Top-1', 30)} | {pad_korean('치명적실수 직후(3수) 회복률', 30)}"
    print(header4)
    print("-" * 105)
    for name, a in [("타겟(분석)", t_adv), ("상대방(평균)", o_adv)]:
        row = f" {pad_korean(name, 12)} | {pad_korean(format_adv_metric(a[0]), 26)} | {pad_korean(format_adv_metric(a[1], True), 30)} | {pad_korean(format_adv_metric(a[2], True), 30)}"
        print(row)

    print_tactical_burst_details([
        ("타겟(분석)", t_burst_score, t_bursts, t_burst_verdict),
        ("상대방(평균)", o_burst_score, o_bursts, o_burst_verdict),
    ])

    print("="*105)
    print(pad_korean("[ 종합 참고 소견 (Reference Verdict) ]", 105))
    print("-" * 105)
    print(f" {pad_korean('진영', 12)} | {pad_korean('전체 AI 유사도', 20)} | {pad_korean('구간 AI 유사도', 20)} | 통계적 분석 소견")
    print("-" * 105)
    print(f" {pad_korean('타겟(분석)', 12)} | {pad_korean(f'{t_fprob:.1f}%', 20)} | {pad_korean(f'{t_pprob:.1f}%', 20)} | {t_verdict}")
    print(f" {pad_korean('상대방(평균)', 12)} | {pad_korean(f'{o_fprob:.1f}%', 20)} | {pad_korean(f'{o_pprob:.1f}%', 20)} | {o_verdict}")
    print("="*105)

    print("\n[ 용어 설명 ]")
    print("■ 비강제수: 누구나 둬야 하는 뻔한 강제수(유일수 등)를 제외하고, 다양한 선택의 여지가 있는 수순입니다.")
    print("■ 난전 Top-1: 국면이 팽팽한 복잡한 상황(1~2순위 수 격차가 적은 경우)에서의 AI 1위수 일치율입니다.")
    print("■ 최고성능구간 ACPL: 비강제수 10수 단위로 쪼갰을 때 가장 점수 손실이 적었던(가장 AI와 유사했던) 구간의 평균 손실입니다.")
    print("■ 어려운 포지션 Top-1: AI 1위수와 2위수의 평가값 격차가 50 이하인 매우 난해한 승부처에서의 1위수 일치율입니다.")
    print("■ 치명적실수 직후 회복률: 200 이상 손해를 본 큰 실수 직후의 다음 3수가 완벽한 방어 수준(손실 10 이하)을 기록한 빈도입니다.")
    print("\n[ AI 유사도 분석 결과 해석 가이드라인 ]")
    print("※ 본 리포트는 기보의 통계적 패턴을 수치화한 '유사도 참고 자료'이며, 단정적인 부정행위의 증거가 될 수 없습니다.")
    print("1. 승부처 일치율(Critical Match): 누구나 맞추는 당연한 수를 제외하고, 복잡한 난전에서 1위 수와")
    print("   일치하는 비율입니다. 최고수 프로 기사도 이 수치가 지속적으로 60%를 넘기기는 매우 어렵습니다.")
    print("2. 스파이크 감지(Spike Detection): 게임 전체 평균은 15~20 수준의 보편적 수치인데, 특정 위기 상황 등")
    print("   특정 10수 구간이나 3~6수 전술 버스트에서 주변 흐름과 다른 정밀도 상승이 반복될 때만 참고 신호로 봅니다.")
    print("3. 난이도 대비 정확도(Complexity Accuracy): 인간은 포지션이 복잡할수록 보통 정확도가 하락하지만,")
    print("   AI는 복잡성에 관계없이 항시 1위 수를 찾아냅니다. 어려운 포지션 일치율이 60%를 상회하면 유의미한 수치입니다.")
    print("4. 비대칭 분포 및 방어 회복: 누적 치명적실수(Blunder)는 0인데 의문수만 유독 많거나, 큰 실수를 한 직후")
    print("   심리적 흔들림 없이 갑자기 완벽한 최선의 수로만 방어한다면, 특이 패턴으로 분류될 수 있습니다.")
    print("5. 기복 및 편차(Variance): 일반적인 인간은 포지션에 따라 25~40 수준의 편차(기복)를 보입니다.")
    print("   만약 편차가 15 미만으로 극도로 일정하며 손실이 적다면, 시스템/AI 특유의 기계적 안정성과 유사합니다.")
    print("="*105)

def print_single_report(cho_moves: List[HistoryEntry], han_moves: List[HistoryEntry], stage_name: str, cho_name: str, han_name: str, use_db: bool, search_config: str) -> None:
    c_stats = calc_stats(cho_moves)
    h_stats = calc_stats(han_moves)
    
    c_adv = calc_advanced_stats(cho_moves)
    h_adv = calc_advanced_stats(han_moves)

    # 1차: 수학 공식 기반 확률 계산
    c_fprob, c_pprob, c_verdict = calculate_cheat_probability(cho_moves)
    h_fprob, h_pprob, h_verdict = calculate_cheat_probability(han_moves)
    c_burst_score, c_bursts, c_burst_verdict = calculate_tactical_burst(cho_moves)
    h_burst_score, h_bursts, h_burst_verdict = calculate_tactical_burst(han_moves)

    # 2차: DB 딥러닝 강제 보정 (Override)
    c_calib_full, c_calib_part, c_db_msg = calibrate_with_db(cho_name, c_fprob, c_pprob, c_stats, use_db, search_config)
    h_calib_full, h_calib_part, h_db_msg = calibrate_with_db(han_name, h_fprob, h_pprob, h_stats, use_db, search_config)

    # DB 보정 후 소견 워딩 재평가
    def re_evaluate_verdict(f_prob, p_prob):
        if p_prob > f_prob + 15.0 and p_prob >= 75.0: return "🔴 특정 난전 구간에서 기계적 수순 전개와 매우 흡사함"
        if f_prob >= 85: return "🔴 기계적 수순 패턴과 매우 높은 통계적 연관성을 보임"
        if f_prob >= 75: return "🟡 인간 최고수 범주를 상회하는 통계적 일치도가 관찰됨"
        if f_prob >= 60: return "🟠 최상위권 프로 및 고수 수준의 정교한 대국 내용"
        return "🟢 보편적인 인간 대국자의 흐름 및 편차"

    if use_db:
        c_verdict = re_evaluate_verdict(c_calib_full, c_calib_part)
        h_verdict = re_evaluate_verdict(h_calib_full, h_calib_part)

    def format_phase(moves, start, end):
        val = get_numeric_phase_acpl(moves, start, end)
        return f"{val:.1f}" if val is not None else "-"

    print("\n" + "="*105)
    print(pad_korean(f"장기 기보 AI 유사도 프로파일링 리포트 (단일 기보) - {stage_name}", 105))
    print("="*105)

    if use_db:
        print("\n 💬 [ 플레이어 성향 상세 분석 (Narrative Report) ]")
        print("-" * 105)
        print(f" ▶ 초(Red) [{cho_name if cho_name else '익명'}]:")
        print(get_playstyle_narrative(cho_name, search_config, cho_moves))
        print(f"\n ▶ 한(Grn) [{han_name if han_name else '익명'}]:")
        print(get_playstyle_narrative(han_name, search_config, han_moves))
        print("-" * 105)

    print(f"\n [ 핵심 지표 (강제수 필터링 적용) ]")
    header = f" {pad_korean('진영', 10)} | {pad_korean('전체ACPL', 11)} | {pad_korean('비강제ACPL', 12)} | {pad_korean('편차(기복)', 11)} | {pad_korean('전체Top-3', 11)} | {pad_korean('비강제Top-1', 12)} | {pad_korean('난전Top-1', 11)}"
    print(header)
    print("-" * 105)
    for name, s in [("초(Red)", c_stats), ("한(Grn)", h_stats)]:
        row = f" {pad_korean(name, 10)} | {s[0]:11.2f} | {s[10]:12.2f} | {s[7]:11.2f} | {pad_korean(f'{s[2]:.1f}%', 11)} | {pad_korean(f'{s[9]:.1f}%', 12)} | {pad_korean(f'{s[8]:.1f}%', 11)}"
        print(row)

    print("\n [ 세분화된 수순 평가 ]")
    header2 = f" {pad_korean('진영', 10)} | {pad_korean('완벽수(≤5cp)', 14)} | {pad_korean('의문수(50~99)', 14)} | {pad_korean('일반실수(100~199)', 14)} | {pad_korean('치명적실수(≥200)', 14)}"
    print(header2)
    print("-" * 105)
    for name, s in [("초(Red)", c_stats), ("한(Grn)", h_stats)]:
        row = f" {pad_korean(name, 10)} | {pad_korean(f'{s[3]} 회', 14)} | {pad_korean(f'{s[4]} 회', 14)} | {pad_korean(f'{s[5]} 회', 14)} | {pad_korean(f'{s[6]} 회', 14)}"
        print(row)

    print("\n [ 구간별 점수 손실 (Phase Analysis) ] *(후반부 AI 평가치 변동 보정용)*")
    header3 = f" {pad_korean('진영', 10)} | {pad_korean('초반 (1~30수)', 25)} | {pad_korean('중반 (31~70수)', 25)} | {pad_korean('종반 (71수 이후)', 25)}"
    print(header3)
    print("-" * 105)
    for name, m in [("초(Red)", cho_moves), ("한(Grn)", han_moves)]:
        row = f" {pad_korean(name, 10)} | {pad_korean(format_phase(m, 0, 29), 25)} | {pad_korean(format_phase(m, 30, 69), 25)} | {pad_korean(format_phase(m, 70, 999), 25)}"
        print(row)

    print("\n [ 특정 구간 AI 밀착도 심층 지표 ]")
    header4 = f" {pad_korean('진영', 10)} | {pad_korean('최고성능구간(10수) ACPL', 26)} | {pad_korean('어려운 포지션(격차<50) Top-1', 30)} | {pad_korean('치명적실수 직후(3수) 회복률', 30)}"
    print(header4)
    print("-" * 105)
    for name, a in [("초(Red)", c_adv), ("한(Grn)", h_adv)]:
        row = f" {pad_korean(name, 10)} | {pad_korean(format_adv_metric(a[0]), 26)} | {pad_korean(format_adv_metric(a[1], True), 30)} | {pad_korean(format_adv_metric(a[2], True), 30)}"
        print(row)

    print_tactical_burst_details([
        ("초(Red)", c_burst_score, c_bursts, c_burst_verdict),
        ("한(Grn)", h_burst_score, h_bursts, h_burst_verdict),
    ])

    print("="*105)
    print(pad_korean("[ 종합 참고 소견 (Reference Verdict) ]", 105))
    print("-" * 105)
    
    if use_db:
        # DB 사용 시: 수학 공식 결과(전체/구간)와 DB 강제 보정 후 결과(전체/구간)를 모두 보여줌
        print(f" {pad_korean('진영', 10)} | {pad_korean('전체 유사도 (수학➔DB보정)', 30)} | {pad_korean('구간 유사도 (수학➔DB보정)', 30)} | 통계적 분석 소견")
        print("-" * 105)
        
        c_full_str = f"{c_fprob:.1f}% ➔ {c_calib_full:.1f}%"
        c_part_str = f"{c_pprob:.1f}% ➔ {c_calib_part:.1f}%"
        print(f" {pad_korean('초(Red)', 10)} | {pad_korean(c_full_str, 30)} | {pad_korean(c_part_str, 30)} | {c_verdict}{c_db_msg}")
        
        h_full_str = f"{h_fprob:.1f}% ➔ {h_calib_full:.1f}%"
        h_part_str = f"{h_pprob:.1f}% ➔ {h_calib_part:.1f}%"
        print(f" {pad_korean('한(Grn)', 10)} | {pad_korean(h_full_str, 30)} | {pad_korean(h_part_str, 30)} | {h_verdict}{h_db_msg}")
    else:
        # DB 미사용 시: 기존 레이아웃 유지
        print(f" {pad_korean('진영', 10)} | {pad_korean('전체 AI 유사도', 20)} | {pad_korean('구간 AI 유사도', 20)} | 통계적 분석 소견")
        print("-" * 105)
        print(f" {pad_korean('초(Red)', 10)} | {pad_korean(f'{c_fprob:.1f}%', 20)} | {pad_korean(f'{c_pprob:.1f}%', 20)} | {c_verdict}")
        print(f" {pad_korean('한(Grn)', 10)} | {pad_korean(f'{h_fprob:.1f}%', 20)} | {pad_korean(f'{h_pprob:.1f}%', 20)} | {h_verdict}")
    print("="*105)
    print("\n[ 용어 설명 ]")
    print("■ 비강제수: 누구나 둬야 하는 뻔한 강제수(유일수 등)를 제외하고, 다양한 선택의 여지가 있는 수순입니다.")
    print("■ 난전 Top-1: 국면이 팽팽한 복잡한 상황(1~2순위 수 격차가 적은 경우)에서의 AI 1위수 일치율입니다.")
    print("■ 최고성능구간 ACPL: 비강제수 10수 단위로 쪼갰을 때 가장 점수 손실이 적었던(가장 AI와 유사했던) 구간의 평균 손실입니다.")
    print("■ 어려운 포지션 Top-1: AI 1위수와 2위수의 평가값 격차가 50 이하인 매우 난해한 승부처에서의 1위수 일치율입니다.")
    print("■ 치명적실수 직후 회복률: 200 이상 손해를 본 큰 실수 직후의 다음 3수가 완벽한 방어 수준(손실 10 이하)을 기록한 빈도입니다.")
    print("\n[ 장기 AI 유사도 판독 가이드라인 (전체 게임) ]")
    print("※ 본 리포트는 기보의 통계적 패턴을 수치화한 '유사도 참고 자료'이며, 단정적인 부정행위의 증거가 될 수 없습니다.")
    print("1. 비강제 ACPL  0 ~ 10 : 매우 기계적이고 완벽한 일치 범위 (Superhuman 범주)")
    print("2. 비강제 ACPL 10 ~ 20 : 최상위 프로 기사 및 고도로 훈련된 전문가 수준")
    print("3. 비강제 ACPL 20 ~ 40 : 일반적인 고수 ~ 아마추어 수준")
    print("4. 후보수 제한이 많은 장기 특성상, 최고수도 Top-3 일치율 70~80%대에 도달할 수 있습니다.")
    print("   다만, 비강제 Top-1 일치율이 70%를 넘거나, 전체 단일 일치율이 90% 이상이라면 AI와의 높은 유사성을 보입니다.")
    print("5. 기복 및 편차(Variance): 일반적인 인간은 포지션에 따라 25~40 수준의 편차(기복)를 보입니다.")
    print("   만약 편차가 15 미만으로 극도로 일정하며 손실이 적다면, 시스템/AI 특유의 기계적 안정성과 유사합니다.")
    print("\n[ 국지적 AI 밀착도 (Partial Match) 패턴 가이드라인 ]")
    print("1. 인간 최고수는 보통 편차(Variance)가 25~40 수준을 기록하지만, AI는 10~20대의 극단적 안정성을 보입니다.")
    print("   기복(편차) 지표가 15 미만으로 이례적으로 낮다면 전반적인 기계적 일관성 여부를 검토해볼 수 있습니다.")
    print("2. 구간별/스파이크 분석에서, 복잡한 '중반(Midgame)'의 손실이 한 자릿수를 기록하거나 특정 10수 구간 또는 3~6수 전술 버스트가")
    print("   주변 흐름보다 뚜렷하게 정밀하다면 난전 구간 검토용 참고 신호로만 해석합니다.")
    print("3. 큰 실수 직후 회복: 치명적인 실수를 한 직후 인간 특유의 연속 실수 없이 갑자기 완벽에 가까운 3수(ACPL<10)를")
    print("   찾아낸다면, 위기 상황에서 외부의 통계적/기계적 도움과 일치하는 흐름으로 해석될 수 있습니다.")
    print("="*105)

def print_final_comparison_summary(all_stages_data: List[dict]):
    print("\n\n" + "!"*135)
    print(pad_korean("★ [ 다중 단계 교차 분석 종합 요약표 (Summary) ] ★", 135))
    print("!"*135)
    
    header = f" {pad_korean('검증 단계', 25)} | {pad_korean('비강제ACPL', 11)} | {pad_korean('비강제Top-1', 11)} | {pad_korean('난전Top-1', 11)} | {pad_korean('편차', 9)} | {pad_korean('최고 유사도', 11)} | 통계적 분석 소견"
    print(header)
    print("-" * 135)

    for data in all_stages_data:
        name = data['name']
        for side_name, moves in data['results'].items():
            s = calc_stats(moves)
            f_p, p_p, verdict = calculate_cheat_probability(moves)
            display_name = f"[{side_name}] {name}"
            row = f" {pad_korean(display_name, 25)} | {s[10]:11.2f} | {pad_korean(f'{s[9]:.1f}%', 11)} | {pad_korean(f'{s[8]:.1f}%', 11)} | {s[7]:9.2f} | {pad_korean(f'{max(f_p, p_p):.1f}%', 11)} | {verdict}"
            print(row)

    print("\n" + "="*135)
    print(pad_korean("[ 교차 분석 판정 핵심 참고사항 ]", 135))
    print("-" * 135)
    print(" 1. 모델 시그니처 감지: 특정 단계에서만 일치율이 60~70% 이상으로 솟구친다면, 해당 계열의 AI와 일치할 가능성이 높습니다.")
    print(" 2. 비강제 ACPL: 초고수 프로 기사도 보통 15~25 수준을 보입니다. 10 이하가 지속된다면 기계적 개입의 강력한 통계적 지표가 됩니다.")
    print(" 3. 국지적 집중: 중반(31~70수) 손실이 한 자릿수이면서 편차가 20 미만인 경우, 결정적인 순간에 매우 높은 AI 일치도를 보인 것입니다.")
    print(" 4. 기복 분석: 대국자는 난전에서 필연적으로 기복(편차 30↑)이 발생합니다. 일관된 낮은 편차는 AI의 전형적인 통계적 특징입니다.")
    print("="*135)

def write_csv_log(filename: str, history: List[HistoryEntry]):
    try:
        with open(filename, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['Game_ID', 'Ply', 'Move', 'Played_UCI', 'Best_UCI', 'Loss(cp)', 'Is_Forced', 'Is_Critical', 'Eval_Gap', 'Rank_Weighted_Match'])
            for h in history:
                writer.writerow([h.game_id, h.ply+1, h.san, h.uci, h.best_move, h.eval_loss, h.is_forced, h.is_critical, h.eval_gap, f'{_rank_weighted_match(h):.2f}'])
        print(f"\n[안내] 상세 분석 로그가 CSV 파일({filename})로 성공적으로 저장되었습니다.")
    except Exception as e:
        print(f"\n[오류] CSV 파일 저장 중 문제가 발생했습니다: {e}")

# ==============================================================================
# MAIN 
# ==============================================================================
def prompt_label_for_anonymous(side_name: str) -> Optional[Tuple[int, str]]:
    print(f"\n[DB 라벨링] 기보에 '{side_name}' 진영의 닉네임이 없습니다.")
    print("이 대국자의 판독 결과를 입력하여 DB 판단 기준을 정밀하게 학습시킬 수 있습니다.")
    print("  1: AI로 확신 (Definite AI)")
    print("  2: AI로 의심 (Suspected AI)")
    print("  3: 사람으로 추정 (Likely Human)")
    print("  4: 사람으로 확신 (Definite Human)")
    print("  엔터: 건너뛰기")
    
    while True:
        try:
            choice = input("선택 (1/2/3/4 또는 엔터): ").strip()
            if not choice: return None
            val = int(choice)
            if val == 1: return (1, "AI 확신")
            elif val == 2: return (2, "AI 의심")
            elif val == 3: return (3, "사람 추정")
            elif val == 4: return (4, "사람 확신")
            print("잘못된 입력입니다. 1~4 사이의 숫자를 입력하세요.")
        except ValueError:
            print("숫자를 입력해주세요.")

def main(argv: Optional[Sequence[str]] = None) -> int:
    init_db()
    
    parser = argparse.ArgumentParser(description="FJACE Profile & Similarity Analyzer v4.3")
    parser.add_argument("inputs", nargs='+', help="Path to PGN files. 'filename:cho' or 'filename:han' targets specific player.")
    parser.add_argument("--engine", default="./stockfish.exe", help="Path to engine")
    parser.add_argument("--depth", type=int, default=15, help="Search depth (기본 15)")
    parser.add_argument("--nodes", type=int, default=None, help="Search nodes")
    parser.add_argument("--nnue", default=None, help="Path to main NNUE file (1단계)")
    parser.add_argument("--nnue2", default=None, help="Path to secondary NNUE file (2단계)")
    parser.add_argument("--threads", type=int, default=1, help="Number of CPU threads")
    
    parser.add_argument("--auto-deep", action="store_true", help="결과가 애매할 경우(60~84%) 현재 엔진에서 자동으로 깊이를 늘려 재탐색")
    parser.add_argument("--deep-add", type=int, default=5, help="--auto-deep 시 추가할 탐색 깊이 (기본 +5)")
    parser.add_argument("--deep-mult", type=float, default=2.0, help="--auto-deep 시 곱할 노드 배수 (기본 x2.0)")
    parser.add_argument("--force-3stage", action="store_true", help="클래식(HCE) 3단계 강제 수행 (기본은 3단계 자동 진행 안 함)")
    
    parser.add_argument("--aux-dir", default=None, help="합의(Consensus) 보조 패널용 앵커 NNUE(2~3개)가 들어있는 폴더")
    parser.add_argument("--fingerprint-dir", default=None, help="여러 NNUE 파일이 들어있는 폴더 경로 지정 (해당 폴더 내 엔진들로 사용 엔진 특정)")
    parser.add_argument("--csv", default=None, help="분석 완료 후 각 수순별 상세 로그를 CSV 파일로 저장 (예: result.csv)")
    
    parser.add_argument("--db", action="store_true", help="DB(janggi_profiles.db) 연동 모드 활성화. 유저 성향 추적 및 다중 라벨링(KNN) 정밀 보정 수행")
    
    args = parser.parse_args(argv)

    engine_path = Path(args.engine)
    if not engine_path.exists():
        raise SystemExit(f"Engine not found: {engine_path}")

    is_single_no_target = (len(args.inputs) == 1 and ":" not in args.inputs[0])
    
    current_search_config = get_search_config(args.depth, args.nodes)

    aux_nnues_for_consensus = []
    if args.aux_dir and Path(args.aux_dir).exists():
        all_aux = list(Path(args.aux_dir).glob("*.nnue"))
        main_nnue_name = Path(args.nnue).name if args.nnue else ""
        valid_aux = [str(p) for p in all_aux if p.name != main_nnue_name]
        aux_nnues_for_consensus = valid_aux[:2] 
        if aux_nnues_for_consensus:
            print(f"\n[알림] {args.aux_dir} 폴더에서 {len(aux_nnues_for_consensus)}개의 앵커 NNUE를 합의(Consensus) 보조 패널로 가동합니다.")
    elif args.nnue2 and Path(args.nnue2).exists():
        aux_nnues_for_consensus = [args.nnue2]
        print(f"\n[알림] --nnue2 에 지정된 {Path(args.nnue2).name} 파일을 합의(Consensus) 단일 보조 패널로 사용합니다.")

    if args.db:
        print(f"\n[DB 모드 활성화] 현재 탐색 환경 '{current_search_config}' 기준으로 DB를 로드/저장합니다.")

    stages = [
        {"num": 1, "nnue": args.nnue, "use": True, "name": "1단계 (최신 NNUE)"},
        {"num": 2, "nnue": args.nnue2, "use": True, "name": "2단계 (보조 NNUE)"},
        {"num": 3, "nnue": None, "use": False, "name": "3단계 (HCE/클래식)"}
    ]

    all_stages_data = []
    stages_run = 0

    cho_all, han_all, target_all, opp_all = [], [], [], []
    white_name, black_name = "", ""

    for stage in stages:
        if stage["num"] == 2 and not args.nnue2: continue
        if stage["num"] == 3 and not args.force_3stage:
            print("\n[안내] 3단계(클래식 분석)는 통계적 기준이 크게 다르므로 기본적으로 생략됩니다. (--force-3stage 옵션으로 강제 실행 가능)")
            break
        
        print(f"\n\n" + "#"*70)
        print(f"### [ {stage['name']} 분석 가동 ]")
        print("#"*70)

        cho_all, han_all, target_all, opp_all, w_n, b_n = run_files_analysis(
            args.inputs, engine_path, args.depth, args.nodes, stage["nnue"], aux_nnues_for_consensus, stage["use"], args.threads, is_single_no_target
        )
        if stage["num"] == 1: white_name, black_name = w_n, b_n

        if is_single_no_target:
            cf, cp, _ = calculate_cheat_probability(cho_all)
            hf, hp, _ = calculate_cheat_probability(han_all)
            max_prob = max(cf, cp, hf, hp)
        else:
            tf, tp, _ = calculate_cheat_probability(target_all)
            max_prob = max(tf, tp)

        if args.auto_deep and 60 <= max_prob < 85:
            print(f"\n\n[안내] 통계적 특이점 구간 감지 (최고 유사도 {max_prob:.1f}%): 현재 엔진에서 심층 재분석을 시도합니다.")
            deep_depth = args.depth + args.deep_add
            deep_nodes = int(args.nodes * args.deep_mult) if args.nodes else None
            
            current_search_config = get_search_config(deep_depth, deep_nodes)
            
            cho_all, han_all, target_all, opp_all, _, _ = run_files_analysis(
                args.inputs, engine_path, deep_depth, deep_nodes, stage["nnue"], aux_nnues_for_consensus, stage["use"], args.threads, is_single_no_target, log_prefix="[심층 탐색] "
            )
            
            if is_single_no_target:
                cf, cp, _ = calculate_cheat_probability(cho_all)
                hf, hp, _ = calculate_cheat_probability(han_all)
                max_prob = max(cf, cp, hf, hp)
            else:
                tf, tp, _ = calculate_cheat_probability(target_all)
                max_prob = max(tf, tp)

        if is_single_no_target:
            print_single_report(cho_all, han_all, stage["name"], white_name, black_name, args.db, current_search_config)
            stage_results = {"name": stage["name"], "results": {"초(Red)": cho_all, "한(Grn)": han_all}}
        else:
            print_target_report(target_all, opp_all, len(args.inputs), stage["name"], args.db, current_search_config)
            stage_results = {"name": stage["name"], "results": {"타겟(분석)": target_all, "상대방(평균)": opp_all}}

        all_stages_data.append(stage_results)
        stages_run += 1
        
        if args.force_3stage:
            print(f"\n[안내] 강제 3단계 모드 활성화: 다음 분석 모델로 교차 검증을 계속합니다.")
            continue

        if stage["num"] == 1:
            if 70 <= max_prob < 85:
                print(f"\n[안내] 보수적 교차 분석: 통계적 유사도가 경계선에 위치하여({max_prob:.1f}%), 2단계(보조 모델) 검증을 추가로 시도합니다.")
            elif max_prob >= 85:
                print(f"\n[안내] AI 패턴과의 유사도가 매우 높게 측정되었습니다({max_prob:.1f}%). 1단계 판독만으로 분석을 조기 종료합니다.")
                break
            else:
                print(f"\n[안내] 판독 결과 일반적인 대국 범주에 부합하여 분석을 조기 종료합니다.")
                break
        elif stage["num"] == 2:
            break

    # [NEW] 터미널 라벨링 피드백 및 DB 누적 파트 (단일 분석 & DB 옵션 사용 시)
    if is_single_no_target and args.db and all_stages_data:
        stage1_data = all_stages_data[0]['results']
        cho_m = stage1_data["초(Red)"]
        han_m = stage1_data["한(Grn)"]
        c_stats, h_stats = calc_stats(cho_m), calc_stats(han_m)

        print("\n" + "="*105)
        print(pad_korean("[ DB 통계 누적 시스템 ]", 105))
        
        if white_name: 
            update_player_profile(white_name, c_stats, [m.uci for m in cho_m[:6]], current_search_config)
        else:
            ans = prompt_label_for_anonymous("초(Red)")
            if ans: update_labeled_stats(ans[0], ans[1], c_stats, current_search_config)
            
        if black_name: 
            update_player_profile(black_name, h_stats, [m.uci for m in han_m[:6]], current_search_config)
        else:
            ans = prompt_label_for_anonymous("한(Grn)")
            if ans: update_labeled_stats(ans[0], ans[1], h_stats, current_search_config)
            
        print("="*105)

    if stages_run > 1:
        print_final_comparison_summary(all_stages_data)
        
    if args.fingerprint_dir and all_stages_data:
        stage1_data = all_stages_data[0]['results']
        if is_single_no_target:
            cho_moves = stage1_data["초(Red)"]
            han_moves = stage1_data["한(Grn)"]
            cho_prob = max(calculate_cheat_probability(cho_moves)[0:2])
            han_prob = max(calculate_cheat_probability(han_moves)[0:2])
            
            fp_targets = []
            if cho_prob >= 60.0: fp_targets.append(("초(Red)", cho_moves))
            if han_prob >= 60.0: fp_targets.append(("한(Grn)", han_moves))
            
            if not fp_targets:
                print("\n[안내] 두 진영 모두 AI 유사도가 비교적 낮아 핑거프린팅 분석의 실효성이 떨어지지만, 분석 요청에 의해 상대적으로 수치가 높은 진영을 1회 검사합니다.")
                if cho_prob >= han_prob: fp_targets.append(("초(Red)", cho_moves))
                else: fp_targets.append(("한(Grn)", han_moves))
                    
            for name, moves in fp_targets:
                run_fingerprinting(engine_path, args.fingerprint_dir, args.depth, args.nodes, args.threads, moves, name)
        else:
            fp_target_moves = stage1_data["타겟(분석)"]
            run_fingerprinting(engine_path, args.fingerprint_dir, args.depth, args.nodes, args.threads, fp_target_moves, "타겟(분석)")
    
    if args.csv:
        if is_single_no_target: all_history = sorted(cho_all + han_all, key=lambda x: x.ply)
        else: all_history = sorted(target_all + opp_all, key=lambda x: x.ply)
        write_csv_log(args.csv, all_history)
        
    return 0

if __name__ == "__main__":
    raise SystemExit(main())