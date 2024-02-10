import flowpy as fp

defaultCoordLabel_channel = lambda label: r"%s/\delta"%label
defaultCoordLabel_blayer = lambda label: r"%s/\theta_0"%label
defaultCoordLabel_pipe = lambda label: r"%s^*"%label

defaultAVGStyle = lambda label: r"\overline{%s}"%label
defaultLocationStyle = lambda x: r"%.3g"%x
defaultTimeStyle = r"t^*"

_style = dict(timeStyle = defaultTimeStyle,
              locationStyle = defaultLocationStyle,
              avgStyle = defaultAVGStyle)

_style['coordLabel'] = defaultCoordLabel_channel
fp.set_default_style(fp.CHANNEL,_style)


_style['coordLabel'] = defaultCoordLabel_blayer
fp.set_default_style(fp.BLAYER,_style)


_style['coordLabel'] = defaultCoordLabel_pipe
fp.set_default_style(fp.PIPE,_style)

_norm_symbols = {'x' : 'x',
                 'z' : 'z',
                 'half-channel':'y',
                 'wall' : 'y^+',
                 'wall_initial':'y^{+0}'}

def get_symbol(mode): return _norm_symbols[mode]
def add_symbols(**kwargs):
    _norm_symbols.update(kwargs)