# coding: utf-8

import math

class Network:

    def __init__(self, input, name=''):
        self.layer = {'n_mul'  : 0, 
                      'n_add'  : 0, 
                      'input'  : input, 
                      'output' : input, 
                      'name'   : name,
                      'child'  : [],
                     }

    def nest(self, dict):
        self.layer['child'].append(dict)
        self.layer['n_mul'] += dict['n_mul']
        self.layer['n_add'] += dict['n_add']
        self.layer['input']  = dict['input']
        self.layer['output'] = dict['output']

        return dict
        
    def del_layers(self, child_idx_begin, child_idx_end=None):

        n_child = len(self.layer['child'])
        child_idx_begin = child_idx_begin if 0 < child_idx_begin else n_child + child_idx_begin
        child_idx_end   = child_idx_end if child_idx_end is not None else n_child-1
        child_idx_end   = child_idx_end if 0 < child_idx_end else n_child + child_idx_end

        for idx in range(child_idx_begin, child_idx_end+1):
            child = self.layer['child'][idx]
            self.layer['n_mul'] -= child['n_mul']
            self.layer['n_add'] -= child['n_add']

        del self.layer['child'][child_idx_begin:child_idx_end+1]

        if 0 < len(self.layer['child']):
            last_child = self.layer['child'][-1]
            self.layer['output'] = last_child['output']

        return self.layer.copy()

    
    def return_dict(self):
        return self.layer.copy()
    

def conv(input, c_out=None, k_size=[3, 3], stride=[1, 1], padding=True):
    input = input['output'] if isinstance(input, dict) else input
    h_in, w_in, c_in = input
    c_out = c_out if c_out is not None else c_in
    h_k, w_k = k_size
    s_x, s_y = stride
    if padding:
        h_out = int(math.ceil(h_in/s_y))
        w_out = int(math.ceil(w_in/s_x))
    else:
        h_out = int(math.ceil((h_in-(h_k-1)+(1-h_k%2))/s_y))
        w_out = int(math.ceil((w_in-(w_k-1)+(1-w_k%2))/s_x))

    n_mul = h_k * w_k * h_out * w_out * c_in * c_out
    n_add = ((h_k * w_k - 1) * h_out * w_out + (c_in - 1)) * c_out \
            + h_out * w_out * c_out # bias

    return {'n_mul'  : n_mul, 
            'n_add'  : n_add, 
            'input'  : input,
            'output' : [h_out, w_out, c_out],
            'name'   : 'convolution',
            }


def bn(input):
    input = input['output'] if isinstance(input, dict) else input
    return {'n_mul'  : input[0]*input[1]*input[2], 
            'n_add'  : input[0]*input[1]*input[2], 
            'input'  : input,
            'output' : input,
            'name'   : 'batch normalization',
            }


def pool(input, k_size=[2, 2], stride=[2, 2], padding=False):
    input = input['output'] if isinstance(input, dict) else input
    h_in, w_in, c_in = input
    h_k, w_k = k_size
    s_x, s_y = stride

    if padding:
        h_out = int(math.ceil(h_in/s_y))
        w_out = int(math.ceil(w_in/s_x))
    else:
        h_out = int(math.ceil((h_in-(h_k-1)+(1-h_k%2))/s_y))
        w_out = int(math.ceil((w_in-(w_k-1)+(1-w_k%2))/s_x))

    return {'n_mul'  : 0, 
            'n_add'  : 0,
            'input'  : [h_in, w_in, c_in],
            'output' : [h_out, w_out, c_in],
            'name'   : 'pooling',
            }


def add(input):
    input = input['output'] if isinstance(input, dict) else input
    return {'n_mul'  : 0, 
            'n_add'  : input[0]*input[1]*input[2], 
            'input'  : input,
            'output' : input,
            'name'   : 'addition',
            }


def concat(dict_list, output=None):
    c = sum([dict['output'][2] for dict in dict_list])
    h, w, _c = output if output is not None else max(dict_list, key=lambda x: x['output'][0]*x['output'][1])['output']
    return {'n_mul'  : 0,
            'n_add'  : 0,
            'input'  : [h, w, c],
            'output' : [h, w, c],
            'name'   : 'concatenate'
            }


def flat(input):
    input = input['output'] if isinstance(input, dict) else input
    return {'n_mul'  : 0, 
            'n_add'  : 0, 
            'input'  : input, 
            'output' : [1, 1, input[0]*input[1]*input[2]],
            'name'   : 'flatten',
            }


def gap(input):
    input = input['output'] if isinstance(input, dict) else input
    return {'n_mul'  : input[2], 
            'n_add'  : (input[0]*input[1]-1)*input[2], 
            'input'  : input, 
            'output' : [1, 1, input[2]],
            'name'   : 'global average pooling',
            }


def fc(input, c_out):
    c_in = input['output'][2] if isinstance(input, dict) else input
    return {'n_mul'  : c_in * c_out, 
            'n_add'  : (c_in - 1) * c_out + c_out, 
            'input'  : [1, 1, c_in], 
            'output' : [1, 1, c_out],
            'name'   : 'full connection',
            }


def d_conv(input, k_size, stride=[1, 1]):
    input = input['output'] if isinstance(input, dict) else input    
    *size, c_out = input
    layer = conv([*size, 1], c_out, k_size, stride)
    layer['name'] = 'depthwise convolution'

    return layer


def p_conv(input, c_out=None, stride=[1, 1]):
    input = input['output'] if isinstance(input, dict) else input    
    layer = conv(input, c_out, k_size=[1, 1], stride=stride)
    layer['name'] = 'pointwise convolution'

    return layer


def sepconv(input, c_out, k_size=[3, 3], stride=[1, 1]):
    input = input['output'] if isinstance(input, dict) else input    
    layer = Network(input, 'separable convolution')
    tmp_dict = layer.nest(d_conv(output, k_size, stride))
    tmp_dict = layer.nest(bn(tmp_dict['output']))
    tmp_dict = layer.nest(p_conv(tmp_dict['output'], c_out, stride=[1, 1]))
    tmp_dict = layer.nest(bn(tmp_dict['output']))

    return layer.return_dict()


def print_madd(dict, print_all=True, indent='  ', cnt=0):
    
    if print_all or cnt==0:
        print(indent*cnt + '- ' + dict['name'])
        print(indent*(cnt+1) + '  n_mul : {:,}'.format(dict['n_mul']))
        print(indent*(cnt+1) + '  n_add : {:,}'.format(dict['n_add']))
        print(indent*(cnt+1) + '  input : ' + str(dict['input']))
        print(indent*(cnt+1) + '  output: ' + str(dict['output']))
        print('')  
    else:
        print(indent*cnt + '- ' + dict['name'])


    if ('child' in dict.keys()):
        if 0 < len(dict['child']) : cnt += 1
        for child in dict['child']: print_madd(child, print_all=print_all, indent=indent, cnt=cnt)        


class AlexNet(Network):

    def __init__(self, input=[227, 227, 3], name='AlexNet'):

        super().__init__(input, name=name)
        temp = self.layer
        temp = self.nest(conv(temp, 96, [11, 11], stride=[4, 4], padding=False))
        temp = self.nest(pool(temp, k_size=[3, 3], stride=[2, 2]))

        temp = self.nest(conv(temp, 256, [5, 5]))
        temp = self.nest(pool(temp, k_size=[3, 3], stride=[2, 2]))

        temp = self.nest(conv(temp, 384))
        temp = self.nest(conv(temp, 384))
        temp = self.nest(conv(temp, 256))
        temp = self.nest(pool(temp, k_size=[3, 3], stride=[2, 2]))
        temp = self.nest(flat(temp))
        temp = self.nest(fc(temp, 4096))
        temp = self.nest(fc(temp, temp['output'][2]))
        temp = self.nest(fc(temp, 1000))


class VGG16(Network):

    def __init__(self, input=[224, 224, 3], name='VGG16'):

        super().__init__(input, name=name)
        temp = self.layer
        temp = self.nest(conv(temp, 64))
        temp = self.nest(conv(temp))
        temp = self.nest(pool(temp))

        temp = self.nest(conv(temp, temp['output'][2]*2))
        temp = self.nest(conv(temp))
        temp = self.nest(pool(temp))

        temp = self.nest(conv(temp, temp['output'][2]*2))
        temp = self.nest(conv(temp))
        temp = self.nest(conv(temp))
        temp = self.nest(pool(temp))

        temp = self.nest(conv(temp, temp['output'][2]*2))
        temp = self.nest(conv(temp))
        temp = self.nest(conv(temp))
        temp = self.nest(pool(temp))

        temp = self.nest(conv(temp))
        temp = self.nest(conv(temp))
        temp = self.nest(conv(temp))
        temp = self.nest(pool(temp))

        temp = self.nest(flat(temp))
        temp = self.nest(fc(temp, 4096))
        temp = self.nest(fc(temp, temp['output'][2]))
        temp = self.nest(fc(temp, 1000))
        

if __name__ == '__main__':

    # calc
    # ignore computations of activation functions, pooolings, concatenate, Non maximum surpression, etc.
    print('\n'+'-'*32+'\n')

    print_madd(AlexNet().return_dict(), print_all=False)

    print('\n'+'-'*32+'\n')

    print_madd(VGG16().return_dict(), print_all=False)

    print('\n'+'-'*32+'\n')

    n_class = 6
    ssd  = VGG16(input=[300, 300, 3], name='Single Shot multibox Detection')
    fm1  = ssd.del_layers(-9)
    temp = ssd.nest(conv(fm1, 4*(n_class+4)))
    o1   = flat(temp)
    
    temp = ssd.nest(pool(fm1))
    temp = ssd.nest(conv(temp, temp['output'][2]*2))
    fm2  = ssd.nest(p_conv(temp))
    temp = ssd.nest(conv(fm2, 6*(n_class+4)))
    o2   = flat(temp)

    temp = ssd.nest(p_conv(fm2, fm2['output'][2]*0.25))
    fm3  = ssd.nest(conv(temp, temp['output'][2]*2, stride=[2,2]))
    temp = ssd.nest(conv(fm3, 6*(n_class+4)))
    o3   = flat(temp)

    temp = ssd.nest(p_conv(fm3, fm3['output'][2]*0.25))
    fm4  = ssd.nest(conv(temp, temp['output'][2]*2, stride=[2,2]))
    temp = ssd.nest(conv(fm4, 6*(n_class+4)))
    o4   = flat(temp)

    temp = ssd.nest(p_conv(fm4, fm4['output'][2]*0.5))
    fm5  = ssd.nest(conv(temp, temp['output'][2]*2, padding=False))
    temp = ssd.nest(conv(fm5, 4*(n_class+4)))
    o5   = flat(temp)

    temp = ssd.nest(p_conv(fm5, fm5['output'][2]*0.5))
    fm6  = ssd.nest(conv(temp, temp['output'][2]*2, padding=False))
    temp = ssd.nest(conv(fm6, 4*(n_class+4)))
    o6   = flat(temp)

    ssd.nest(concat([o1, o2, o3, o4, o5, o6]))

    print_madd(ssd.return_dict(), print_all=True)

    print('\n'+'-'*32+'\n')

