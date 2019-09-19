import numpy as np 
from collections import Counter

class Color():
    HEADER    = '\033[95m'
    OKBLUE    = '\033[94m'
    OKGREEN   = '\033[92m'
    WARNING   = '\033[93m'
    LIGHTBLUE = '\033[96m'
    FADE      = '\033[90m'
    FAIL      = '\033[91m'
    ENDC      = '\033[0m'
    BOLD      = '\033[1m'
    UNDERLINE = '\033[4m'
    
    @staticmethod
    def flt(flt):
        r = "%.4f" % flt
        return f"{Color.FADE}{r}{Color.ENDC}"
    
    @staticmethod
    def bold(txt):
        return f"{Color.OKGREEN}{txt}{Color.ENDC}"
    
    @staticmethod
    def val(flt, **kwargs):
        c = kwargs.get('color', 'yellow')
        e = kwargs.get('extra', '')
        f = kwargs.get('f', 4)

        if flt != float('-inf'):
            r = f"%.{f}f" % flt if flt != None else None
        else:
            r = '-∞'
            
        colors = {
            'yellow': Color.WARNING,
            'blue':  Color.OKBLUE,
            'orange': Color.FAIL,
            'green': Color.OKGREEN,
            'lightblue': Color.LIGHTBLUE
        }
        return f"{colors.get(c)}{e}{r}{Color.ENDC}"
        
class Mixture:
    def __init__(self, **kwargs):
        self.maxs      = kwargs['maxs']
        self.mins      = kwargs['mins']
        self.deltas    = dict.get(kwargs, 'deltas', [])
        self.spreads   = self.maxs - self.mins
        self.dimension = dict.get(kwargs, 'dimension', None)
        self.children  = dict.get(kwargs, 'children', [])
        self.depth     = dict.get(kwargs, 'depth', 0)
        self.n         = kwargs['n']
        self.parent    = dict.get(kwargs, 'parent', None)
        self.splits    = dict.get(kwargs, 'splits', []) # for bins algo

        #assert np.all(self.spreads > 0)

    def __repr__(self, level = 0):
        _dim = Color.val(self.dimension, f=0, color='orange', extra="dim=")
        _dep = Color.val(self.depth, f=0, color='yellow', extra="dep=")
        _nnn = Color.val(self.n, f=0, color='green', extra="n=")
        _rng = [f"{round(self.mins[i],2)} - {round(self.maxs[i],2)}" for i, _ in enumerate(self.mins)]
        _rng = ", ".join(_rng)
        
        if self.mins.shape[0] > 4:
            _rng = "..."

        _sel = " "*(level) + f"✚ Mixture [{_rng}] {_dim} {_dep} {_nnn}"

        if level <= 100:
            for split in self.children:
                _sel += f"\n{split.__repr__(level+2)}"
        else:
            _sel += " ..."
        return f"{_sel}"

class Separator:
    def __init__(self, **kwargs):
        self.split     = kwargs['split']
        self.dimension = kwargs['dimension']
        self.depth     = kwargs['depth']
        self.children  = kwargs['children']
        self.parent    = kwargs['parent']

    def __repr__(self, level=0):
        _sel = " "*(level) + f"ⓛ Separator dim={self.dimension} split={round(self.split, 2)}"
        
        for child in self.children:
            _sel += f"\n{child.__repr__(level+2)}"

        return _sel

class GPMixture:
    def __init__(self, **kwargs):
        self.mins      = kwargs['mins']
        self.maxs      = kwargs['maxs']
        self.idx       = dict.get(kwargs, 'idx', [])
        self.parent    = kwargs['parent']

    def __repr__(self, level = 0):
        _rng = [f"{round(self.mins[i],2)} - {round(self.maxs[i],2)}" for i, _ in enumerate(self.mins)]
        _rng = ", ".join(_rng)
        if self.mins.shape[0] > 4:
            _rng = "..."

        return " "*(level) + f"⚄ GPMixture [{_rng}] n={len(self.idx)}"

def _cached_gp(cache, **kwargs):
    min_, max_ = list(kwargs['mins']), list(kwargs['maxs'])
    cached = dict.get(cache, (*min_, *max_))
    if not cached:
        cache[(*min_,*max_)] = GPMixture(**kwargs)
        
    return cache[(*min_,*max_)]

def query(X, mins, maxs, skipleft=False):
    mask, D = np.full(len(X), True), X.shape[1]
    for d_ in range(D):
        if not skipleft:
            mask = mask & (X[:, d_] >= mins[d_]) & (X[:,d_] <= maxs[d_])
        else:
            mask = mask & (X[:, d_] >= mins[d_]) & (X[:,d_] <= maxs[d_])
    return np.nonzero(mask)[0]

def build(**kwargs):
    X             = kwargs['X']
    mins, maxs    = np.min(X, 0), np.max(X, 0)
    max_depth     = dict.get(kwargs, 'max_depth',       8)
    min_samples   = dict.get(kwargs, 'min_samples',     1)
    max_samples   = dict.get(kwargs, 'max_samples', 10**4)
    delta_divisor = dict.get(kwargs, 'delta_divisor',   2) 

    if dict.get(kwargs, 'deltas') is not None:
        deltas = kwargs['deltas']
    else:
        deltas = (maxs - mins) / delta_divisor
    
    root_node          = Mixture(mins = mins, maxs=maxs, deltas=deltas, n=len(X), parent=None)
    to_process, cache  = [root_node], dict()
    nsplits            = Counter()
    
    while len(to_process):
        node = to_process.pop()
        
        if type(node) is not Mixture:
            continue

        if node.dimension == None:
            node.dimension = node.depth #0 #np.random.randint(0, D)
            
        d             = node.dimension
        spread, delta = node.spreads[d], node.deltas[d]
        min_, max_    = node.mins[d], node.maxs[d]
        n_splits, _   = divmod(spread, delta)
        splits        = min_ + delta * np.arange(1, n_splits + 1)
    
        if len(splits) and np.isclose(splits[-1], max_):
            splits = splits[0:-1]

        if not len(splits):
            raise Exception('1')
            idx = query(X, node.mins, node.maxs)
            gp  = _cached_gp(cache, mins=node.mins, maxs=node.maxs, idx=idx, parent=node)
            node.children.append(gp)
            continue

        nsplits[len(splits)] += 1

        for split in splits:
            create_mixtures = (node.spreads >= (2 * node.deltas))

            create_mixtures_left  = create_mixtures.copy()
            create_mixtures_right = create_mixtures.copy()

            create_mixtures_left[d]  = split - node.mins[d] >= (2*node.deltas[d])
            create_mixtures_right[d] = node.maxs[d] - split >= (2*node.deltas[d])

            n_max, n_min       = node.maxs.copy(), node.mins.copy()
            n_max[d], n_min[d] = split, split
            
            idx_left   = query(X, node.mins, n_max,     skipleft=False)
            idx_right  = query(X, n_min,     node.maxs, skipleft=True)
            next_depth = node.depth+1
            opts_left = {
                'mins':      node.mins,
                'maxs':      n_max, 
                'deltas':    node.deltas, 
                'depth':     next_depth, 
                'dimension': np.argmax(create_mixtures_left),
                'n':         len(idx_left),
                'parent':    node
            }
            if np.any(create_mixtures_left) and len(idx_left) >= min_samples and next_depth < max_depth:
                left = Mixture(**opts_left)
            elif len(idx_left) > max_samples:
                print(f"Forced a mixture. {len(idx_left)}")
                left = Mixture(**opts_left)
            else:
                left = _cached_gp(cache, mins=node.mins, maxs=n_max, idx=idx_left, parent=node)
            
            opts_right = {
                'mins':      n_min, 
                'maxs':      node.maxs, 
                'deltas':    node.deltas, 
                'depth':     next_depth, 
                'dimension': np.argmax(create_mixtures_right),
                'n':         len(idx_right),
                'parent':    node
            }

            if np.any(create_mixtures_right) and len(idx_right) >= min_samples and next_depth < max_depth:
                right = Mixture(**opts_right)
            elif len(idx_right) > max_samples:
                print(f"Forced a mixture. {len(idx_right)}")
                right = Mixture(**opts_right)
            else:
                right = _cached_gp(cache, mins=n_min, maxs=node.maxs, idx=idx_right, parent=node)
            
            # the trick
            if len(idx_right) < min_samples:
                #create_mixtures_right[opts_left['dimension']] = 0
                # opts_left['maxs'] = node.maxs
                # opts_left['dimension'] = opts_left['dimension']+1

                # m = Mixture(**opts_left)
                # node.children.append(m)
                # to_process.append(m)

                node.children.append(left)
                to_process.append(left)
                continue
            
            if len(idx_left) < min_samples:
                #create_mixtures_left[opts_right['dimension']] = 0
                # opts_right['mins'] = node.mins
                # opts_right['dimension'] = opts_right['dimension']+1

                # m = Mixture(**opts_right)
                # node.children.append(m)
                # to_process.append(m)
                node.children.append(right)
                to_process.append(right)
                continue
            
            to_process.extend([left, right])
            separator_opts = {
                'dimension':             d, 
                'split':             split, 
                'children':  [left, right], 
                'parent':             None,
                'depth':        node.depth
            }
            node.children.append(Separator(**separator_opts))

    gps = list(cache.values())
    aaa = [len(gp.idx) for gp in gps]
    c = (np.mean(aaa)**3)*len(aaa)
    r = 1-(c/(len(X)**3))
    print("Full:\t\t", len(X)**3, "\nOptimized:\t", int(c), f"\n#GP's:\t\t {len(gps)} ({np.min(aaa)}-{np.max(aaa)})", "\nReduction:\t", f"{round(100*r, 4)}%")
    print(f"nsplits:\t {nsplits}")
    print(f"Lengths:\t {aaa}\nSum:\t\t {sum(aaa)} (N={len(X)})")
    return root_node, gps

def get_splits(X, dd, **kwargs):
    meta      = dict.get(kwargs, 'meta', [""] * X.shape[1])
    max_depth = dict.get(kwargs, 'max_depth', 8)
    log       = dict.get(kwargs, 'log', False)
    
    features_mask = np.zeros(X.shape[1])
    splits        = np.zeros((X.shape[1], dd-1))
    quantiles     = np.quantile(X, np.arange(0, 1, 1/dd)[1:], axis=0).T
    for i, var in enumerate(quantiles):
        include = False
        if dd == 2:
            spread = np.sum(X[:, i] < var[0]) - np.sum(X[:, i] >= var[0])
            
            if np.abs(spread) < X.shape[0]/12:
                include = True
        elif len(np.unique(np.round(var, 8))) == len(var):
                include = True
        
        if include:
            features_mask[i] = 1
            splits[i]        = np.array(var)

            if np.sum(features_mask) <= max_depth and meta and log: 
                print(i, "\t", meta[i], var)
            else: pass #print('.', end = '')

    return splits, features_mask

def build_bins(**kwargs):
    X                   = kwargs['X']
    max_depth           = dict.get(kwargs, 'max_depth',               8)
    min_samples         = dict.get(kwargs, 'min_samples',             0)
    max_samples         = dict.get(kwargs, 'max_samples',         10**4)
    qd                  = dict.get(kwargs, 'qd',                      0)
    log                 = dict.get(kwargs, 'log',                 False)
    jump                = dict.get(kwargs, 'jump',                False)
    reduce_branching    = dict.get(kwargs, 'reduce_branching',    False)
    randomize_branching = dict.get(kwargs, 'randomize_branching', False)
    
    mins, maxs       = np.min(X, 0), np.max(X, 0)

    splits, features_mask = get_splits(X, qd, meta=dict.get(kwargs, 'meta', None), log=log)
    
    root_mixture_opts = {
        'mins':      np.min(X, 0), 
        'maxs':      np.max(X, 0), 
        'n':         len(X), 
        'parent':    None,
        'dimension': np.argmax(features_mask)
    }

    nsplits            = Counter()
    root_node          = Mixture(**root_mixture_opts)
    to_process, cache  = [root_node], dict()

    while len(to_process):
        node = to_process.pop()
        
        if type(node) is not Mixture:
            continue
        
        d = node.dimension
        fit_lhs = node.mins < splits[:,  0]
        fit_rhs = node.maxs > splits[:, -1]
        create  = np.logical_and(fit_lhs, fit_rhs)
        create  = np.logical_and(create, features_mask)
    
        # Preprocess splits
        node_splits = []
        for node_split in splits[d]:
            # We skip the split completely if it is outside of 
            # the scope of the data in this dimension. parent split 
            # has the data already. this saves us form n = 0 mixtures
            if node_split <= node.mins[d] or node_split >= node.maxs[d]:
                continue

            node_splits.append(node_split)
        
        # Jumping results in high branching of new Mixtures, to 
        # help this problem we reduce the creation of new branches
        # for depth >= 1
        if reduce_branching and node.depth >= 1:
            node_splits = [np.median(node_splits)]

        if len(node_splits) == 0: raise Exception('1')
        
        for split in node_splits:
            create_left  = create.copy()
            create_right = create.copy()

            create_left[d]  = split != node_splits[0] 
            create_right[d] = split != node_splits[-1]

            if jump:
                # We force a new dimension for every child 
                # on the same split level
                create_left[d], create_right[d]      = False, False
                create_right[np.argmax(create_left)] = False
            else:
                # We dont create new mixture in the limits
                create_left[d]  = split != node_splits[0] 
                create_right[d] = split != node_splits[-1]
    
            new_maxs, new_mins       = node.maxs.copy(), node.mins.copy()
            new_maxs[d], new_mins[d] = split, split
            
            idx_left   = query(X, node.mins, new_maxs,  skipleft=False)
            idx_right  = query(X, new_mins,  node.maxs, skipleft=True)
            
            next_depth = node.depth+1

            loop = [
                ('left',  create_left,  idx_left,  node.mins, new_maxs),
                ('right', create_right, idx_right, new_mins, node.maxs)
            ]
            
            results = []
            for _, create_mixture, idx, mins, maxs, in loop:
                if min_samples == 0:
                    min_samples = min(len(idx_left), len(idx_right)) + 1
                
                can_create     = np.any(create_mixture)
                big_enough     = len(idx)   >= min_samples
                not_too_big    = len(idx)   <= max_samples
                not_too_deep   = next_depth  < max_depth

                if randomize_branching:
                    next_dimension = np.random.choice(np.where(create_mixture)[0])
                else:
                    next_dimension = np.argmax(create_mixture)

                mixture_opts = {
                        'mins':      mins,
                        'maxs':      maxs,
                        'depth':     next_depth,
                        'dimension': next_dimension,
                        'n':         len(idx)
                }

                if all([can_create, big_enough, not_too_deep, not_too_big]):    
                    results.append(Mixture(**mixture_opts))
                elif can_create and len(idx) > max_samples:
                    print("Forcing a Mixture...")
                    results.append(Mixture(**mixture_opts))
                elif len(idx):
                    if len(idx) > max_samples:
                        print(f"Had to create a GP with n={len(idx)} because we ran out of splits.")
                    gp = _cached_gp(cache, mins=mins, maxs=maxs, idx=idx, parent=None)
                    results.append(gp)

            if len(results) == 2:
                to_process.extend(results)
                separator_opts = {
                    'depth': node.depth,
                    'dimension':      d, 
                    'split':      split, 
                    'children': results, 
                    'parent':       None
                }
                node.children.append(Separator(**separator_opts))
            elif len(results) == 1:
                node.children.extend(results)
                to_process.extend(results)
            else:
                raise Exception('1')

    gps = list(cache.values())
    aaa = [len(gp.idx) for gp in gps]
    c = (np.mean(aaa)**3)*len(aaa)
    r = 1-(c/(len(X)**3))
    print("Full:\t\t", len(X)**3, "\nOptimized:\t", int(c), f"\n#GP's:\t\t {len(gps)} ({np.min(aaa)}-{np.max(aaa)})", "\nReduction:\t", f"{round(100*r, 4)}%")
    print(f"nsplits:\t {nsplits}")
    print(f"Lengths:\t {aaa}\nSum:\t\t {sum(aaa)} (N={len(X)})")

    return root_node, gps