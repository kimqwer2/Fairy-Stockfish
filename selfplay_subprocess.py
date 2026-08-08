#!/usr/bin/env python3
import argparse, subprocess, math, random, os

PASS_MOVES = {"0000"}
NO_LEGAL_MOVES = {"(none)", "none"}

def read_epd(path):
    fens=[]
    with open(path) as f:
        for line in f:
            s=line.strip()
            if not s or s.startswith('#'):
                continue
            fen=s.split(';')[0].strip()
            fens.append(fen)
    return fens

class Engine:
    def __init__(self,path,variant,nnue,movetime,depth=None,extra_opts=None):
        self.p=subprocess.Popen([path],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True,bufsize=1)
        self.movetime=movetime
        self.send('uci'); self.wait('uciok')
        self.setopt('Threads','1'); self.setopt('Hash','16'); self.setopt('Use NNUE','true')
        if nnue:
            if not os.path.exists(nnue):
                raise RuntimeError(f'NNUE file not found: {nnue}')
            self.setopt('EvalFile',nnue)
        self.setopt('UCI_Variant',variant)
        self.depth=depth
        if extra_opts:
            for k,v in extra_opts.items():
                self.setopt(k,v)
        self.send('isready'); self.wait('readyok')
        if nnue:
            print(f'NNUE LOADED: YES ({path} -> {nnue})')

    def send(self,s):
        self.p.stdin.write(s+'\n')
        self.p.stdin.flush()

    def wait(self,token):
        while True:
            l=self.p.stdout.readline()
            if not l:
                raise RuntimeError('engine died')
            if token in l:
                return l.strip()

    def bestmove(self,fen,moves):
        cmd='position fen '+fen
        if moves:
            cmd += ' moves ' + ' '.join(moves)
        self.send(cmd)
        self.send(f'go depth {self.depth}' if self.depth is not None else f'go movetime {self.movetime}')
        while True:
            l=self.p.stdout.readline()
            if not l:
                raise RuntimeError('engine died')
            if l.startswith('bestmove '):
                return l.split()[1]

    def bestmove_startpos(self,moves):
        cmd='position startpos'
        if moves:
            cmd += ' moves ' + ' '.join(moves)
        self.send(cmd)
        self.send(f'go depth {self.depth}' if self.depth is not None else f'go movetime {self.movetime}')
        while True:
            l=self.p.stdout.readline()
            if not l:
                raise RuntimeError('engine died')
            if l.startswith('bestmove '):
                return l.split()[1]

    def eval_cp(self, fen):
        self.send('position fen '+fen)
        self.send('go depth 8')
        last_cp = None
        while True:
            l=self.p.stdout.readline()
            if not l:
                raise RuntimeError('engine died')
            if l.startswith('info ') and ' score cp ' in l:
                try:
                    last_cp = int(l.split(' score cp ')[1].split()[0])
                except Exception:
                    pass
            if l.startswith('bestmove '):
                return last_cp

    def setopt(self,n,v):
        self.send(f'setoption name {n} value {v}')

    def close(self):
        try:
            self.send('quit')
        except Exception:
            pass
        self.p.terminate()

def elo(score):
    score=max(1e-6,min(1-1e-6,score))
    return -400*math.log10(1/score-1)

def play_game(wa,ba,fen,maxplies,debug=False):
    moves=[]
    consecutive_passes = 0
    for ply in range(maxplies):
        stm = 'White' if ply % 2 == 0 else 'Black'
        eng=wa if ply%2==0 else ba
        bm=eng.bestmove(fen,moves)

        if bm in NO_LEGAL_MOVES:
            winner = 0 if ply % 2 == 0 else 1
            if debug:
                print(f'DEBUG terminal=no_legal_move stm={stm} ply={ply} winner={"White" if winner==1 else "Black"} score={winner}')
            return winner, 'no_move'

        if bm in PASS_MOVES:
            consecutive_passes += 1
            moves.append(bm)
            if debug:
                print(f'DEBUG pass_move stm={stm} ply={ply} move={bm} consecutive_passes={consecutive_passes}')
            if consecutive_passes >= 2:
                if debug:
                    print('DEBUG terminal=double_pass score=0.5')
                return 0.5, 'double_pass'
            continue

        consecutive_passes = 0
        moves.append(bm)

    return 0.5, 'maxplies'

def generate_positions(args):
    rng = random.Random(args.seed)
    eng = Engine(args.engine, args.variant, args.nnue, args.movetime, args.depth)
    try:
        out = []
        seen = set()
        attempts = 0
        while len(out) < args.count and attempts < args.count * args.max_retries:
            attempts += 1
            moves = []
            plies = rng.randint(args.min_plies, args.max_plies)
            for _ in range(plies):
                eng.setopt('Skill Level', str(rng.randint(args.skill_min, args.skill_max)))
                bm = eng.bestmove_startpos(moves) if args.use_startpos else eng.bestmove(args.start_fen, moves)
                if bm in NO_LEGAL_MOVES:
                    break
                moves.append(bm)
            if not moves:
                continue

            if args.use_startpos:
                eng.send('position startpos' + (' moves ' + ' '.join(moves) if moves else ''))
            else:
                eng.send('position fen ' + args.start_fen + (' moves ' + ' '.join(moves) if moves else ''))
            eng.send('d')
            fen = None
            while True:
                line = eng.p.stdout.readline()
                if not line:
                    break
                if line.startswith('Fen: '):
                    fen = line[len('Fen: '):].strip()
                if (line.strip() == '' or line.startswith('Checkers:')) and fen:
                    break

            if not fen or fen in seen:
                continue

            cp = eng.eval_cp(fen)
            if cp is None or abs(cp) < args.min_imbalance_cp or abs(cp) > args.max_eval_cp:
                continue

            seen.add(fen)
            out.append(fen)

        if len(out) < args.count:
            raise RuntimeError(f'Generated only {len(out)} positions out of requested {args.count}')

        with open(args.output, 'w') as f:
            for fen in out:
                f.write(fen + '\n')

        print(f'Generated {len(out)} positions -> {args.output}')
    finally:
        eng.close()

def run_match(args):
    fens=read_epd(args.epd)
    copt={}
    bopt={}
    if args.cand_skill is not None: copt['Skill Level']=str(args.cand_skill)
    if args.base_skill is not None: bopt['Skill Level']=str(args.base_skill)
    base_mt = args.base_movetime if args.base_movetime is not None else args.movetime
    cand_mt = args.cand_movetime if args.cand_movetime is not None else args.movetime
    base=Engine(args.base,args.variant,args.nnue,base_mt,args.base_depth,bopt)
    cand=Engine(args.cand,args.variant,args.nnue,cand_mt,args.cand_depth,copt)
    print(f'Candidate engine: {args.cand}')
    print(f'Baseline engine:  {args.base}')
    w=d=l=0
    ww=dw=lw=0
    wb=db=lb=0
    reason_counts = {'maxplies': 0, 'no_move': 0, 'abnormal': 0, 'double_pass': 0}
    try:
        for g in range(args.games):
            fen=fens[(g // 2) % len(fens)]
            try:
                if g%2==0:
                    cand_side='White'
                    r, reason = play_game(cand,base,fen,args.maxplies,args.debug)
                else:
                    cand_side='Black'
                    r, reason = play_game(base,cand,fen,args.maxplies,args.debug)
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
            except Exception:
                reason_counts['abnormal'] += 1
                r = 0.5
                reason = 'abnormal'

            cand_score = r if g % 2 == 0 else (1-r if r in (0,1) else r)
            if cand_score==1: w+=1
            elif cand_score==0: l+=1
            else: d+=1

            if g % 2 == 0:
                if cand_score==1: ww+=1
                elif cand_score==0: lw+=1
                else: dw+=1
            else:
                if cand_score==1: wb+=1
                elif cand_score==0: lb+=1
                else: db+=1

            outcome='Draw' if cand_score==0.5 else ('Candidate win' if cand_score==1 else 'Candidate loss')
            print(f'game {g+1}: candidate {cand_side} -> {outcome} ({reason})')
            if (g+1)%10==0: print(f'games={g+1} W/D/L={w}/{d}/{l}')
    finally:
        base.close(); cand.close()

    total=w+d+l
    score=(w+0.5*d)/total
    print(f'Total {total} W {w} D {d} L {l} Score {score:.3f} Elo {elo(score):.1f}')
    wt = ww + dw + lw
    bt = wb + db + lb
    if wt:
        print(f'Candidate as White: W {ww} D {dw} L {lw} Score {(ww + 0.5 * dw) / wt:.3f}')
    if bt:
        print(f'Candidate as Black: W {wb} D {db} L {lb} Score {(wb + 0.5 * db) / bt:.3f}')
    print('Termination breakdown:')
    print(f'  no_move: {reason_counts.get("no_move", 0)}')
    print(f'  double_pass: {reason_counts.get("double_pass", 0)}')
    print(f'  maxplies: {reason_counts.get("maxplies", 0)}')
    print(f'  abnormal: {reason_counts.get("abnormal", 0)}')

def main():
    ap=argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='cmd', required=True)
    m = sp.add_parser('match')
    m.add_argument('--base',required=True)
    m.add_argument('--cand',required=True)
    m.add_argument('--epd',default='janggimodern.epd')
    m.add_argument('--variant',default='janggimodern')
    m.add_argument('--nnue',default='janggimodern-18.nnue')
    m.add_argument('--games',type=int,default=80)
    m.add_argument('--movetime',type=int,default=30)
    m.add_argument('--maxplies',type=int,default=220)
    m.add_argument('--base-movetime',type=int)
    m.add_argument('--cand-movetime',type=int)
    m.add_argument('--cand-skill',type=int)
    m.add_argument('--base-skill',type=int)
    m.add_argument('--base-depth',type=int)
    m.add_argument('--cand-depth',type=int)
    m.add_argument('--debug', action='store_true')

    g = sp.add_parser('gen')
    g.add_argument('--engine', required=True)
    g.add_argument('--output', default='janggimodern_midgame.epd')
    g.add_argument('--variant', default='janggimodern')
    g.add_argument('--nnue', default='janggimodern-18.nnue')
    g.add_argument('--count', type=int, default=96)
    g.add_argument('--min-plies', type=int, default=20)
    g.add_argument('--max-plies', type=int, default=60)
    g.add_argument('--movetime', type=int, default=35)
    g.add_argument('--depth', type=int)
    g.add_argument('--max-retries', type=int, default=25)
    g.add_argument('--seed', type=int, default=1)
    g.add_argument('--skill-min', type=int, default=-20)
    g.add_argument('--skill-max', type=int, default=20)
    g.add_argument('--min-imbalance-cp', type=int, default=35)
    g.add_argument('--max-eval-cp', type=int, default=700)
    g.add_argument('--start-fen', default='rheagaehr/9/1c5c1/s1s1s1s1s/9/9/S1S1S1S1S/1C5C1/9/RHEAGAEHR w - - 0 1')
    g.add_argument('--use-startpos', action='store_true', default=True)

    args=ap.parse_args()
    if args.cmd == 'gen':
        generate_positions(args)
    else:
        run_match(args)

if __name__=='__main__':
    main()
