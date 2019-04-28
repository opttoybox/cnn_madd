# coding: utf-8

import math

class Network:

    def __init__(self, shape, name=''):
        self.ret_dict = {'n_mul' : 0, 
                        'n_add' : 0, 
                        'input' : shape, 
                        'shape' : shape, 
                        'name'  : name,
                        'child' : [],
                        }

    def nest(self, dict):
        self.ret_dict['child'].append(dict)
        self.ret_dict['n_mul'] += dict['n_mul']
        self.ret_dict['n_add'] += dict['n_add']
        self.ret_dict['shape']  = dict['shape']

        return dict
        
    def del_layers(self, child_idx_begin, child_idx_end=None):

        n_child = len(self.ret_dict['child'])
        child_idx_begin = child_idx_begin if 0 < child_idx_begin else n_child + child_idx_begin
        child_idx_end   = child_idx_end if child_idx_end is not None else n_child-1
        child_idx_end   = child_idx_end if 0 < child_idx_end else n_child + child_idx_end

        for idx in range(child_idx_begin, child_idx_end+1):
            child = self.ret_dict['child'][idx]
            self.ret_dict['n_mul'] -= child['n_mul']
            self.ret_dict['n_add'] -= child['n_add']

        del self.ret_dict['child'][child_idx_begin:child_idx_end+1]

        if 0 < len(self.ret_dict['child']):
            last_child = self.ret_dict['child'][-1]
            self.ret_dict['shape'] = last_child['shape']

        return self.ret_dict

    
    def return_dict(self):
        return self.ret_dict
    

def conv(shape, c_out, k_size=[3, 3], stride=[1, 1], padding=True):

    h_in, w_in, c_in = shape
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
            'shape'  : [h_out, w_out, c_out],
            'name'   : 'convolution',
            }


def bn(shape):
    return {'n_mul'  : shape[0]*shape[1]*shape[2], 
            'n_add'  : shape[0]*shape[1]*shape[2], 
            'shape'  : shape,
            'name'   : 'batch normalization',
            }


def pool(shape, k_size=[2, 2], stride=[2, 2], padding=False):
    
    h_in, w_in, c_in = shape
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
            'shape'  : [h_out, w_out, c_in],
            'name'   : 'pooling',
            }


def add(shape):
    return {'n_mul'  : 0, 
            'n_add'  : shape[0]*shape[1]*shape[2], 
            'shape'  : shape,
            'name'   : 'addition',
            }


def flat(shape):
    return {'n_mul'  : 0, 
            'n_add'  : 0, 
            'shape'  : [1, 1, shape[0]*shape[1]*shape[2]],
            'name'   : 'flatten',
            }


def gap(shape):
    return {'n_mul'  : shape[2], 
            'n_add'  : (shape[0]*shape[1]-1)*shape[2], 
            'shape'  : [1, 1, shape[2]],
            'name'   : 'global average pooling',
            }


def fc(c_in, c_out):
    return {'n_mul' : c_in * c_out, 
            'n_add' : (c_in - 1) * c_out + c_out, 
            'shape' : [1, 1, c_out],
            'name'  : 'full connection',
            }


def d_conv(shape, k_size, stride=[1, 1]):
    
    *size, c_out = shape
    ret_dict = conv([*size, 1], c_out, k_size, stride)
    ret_dict['name'] = 'depthwise convolution'

    return ret_dict


def p_conv(shape, c_out, stride=[1, 1]):
    
    ret_dict = conv(shape, c_out, k_size=[1, 1], stride=stride)
    ret_dict['name'] = 'pointwise convolution'

    return ret_dict


def sepconv(shape, c_out, k_size=[3, 3], stride=[1, 1]):
    
    ret_dict = Network('separable convolution')
    tmp_dict = ret_dict.nest(d_conv(shape, k_size, stride))
    tmp_dict = ret_dict.nest(bn(tmp_dict['shape']))
    tmp_dict = ret_dict.nest(p_conv(tmp_dict['shape'], c_out, stride=[1, 1]))
    tmp_dict = ret_dict.nest(bn(tmp_dict['shape']))

    return ret_dict.return_dict()


def print_madd(dict, print_tree=True, indent='  ', cnt=0):
    
    print(indent*cnt + '- ' + dict['name'])
    print(indent*(cnt+1) + '  n_mul: {:,}'.format(dict['n_mul']))
    print(indent*(cnt+1) + '  n_add: {:,}'.format(dict['n_add']))
    print(indent*(cnt+1) + '  shape: ' + str(dict['shape']))
    print('')  

    if ('child' in dict.keys()) and print_tree:
        if 0 < len(dict['child']) : cnt += 1
        for child in dict['child']: print_madd(child, print_tree=print_tree, indent=indent, cnt=cnt)        


class AlexNet(Network):

    def __init__(self, shape=[227, 227, 3], name='AlexNet'):

        super().__init__(shape, name=name)
        temp = self.ret_dict
        temp = self.nest(conv(temp['shape'], 96, [11, 11], stride=[4, 4], padding=False))
        temp = self.nest(pool(temp['shape'], k_size=[3, 3], stride=[2, 2]))

        temp = self.nest(conv(temp['shape'], 256, [5, 5]))
        temp = self.nest(pool(temp['shape'], k_size=[3, 3], stride=[2, 2]))

        temp = self.nest(conv(temp['shape'], 384))
        temp = self.nest(conv(temp['shape'], 384))
        temp = self.nest(conv(temp['shape'], 256))
        temp = self.nest(pool(temp['shape'], k_size=[3, 3], stride=[2, 2]))
        temp = self.nest(flat(temp['shape']))
        temp = self.nest(fc(temp['shape'][2], 4096))
        temp = self.nest(fc(temp['shape'][2], temp['shape'][2]))
        temp = self.nest(fc(temp['shape'][2], 1000))


class VGG16(Network):

    def __init__(self, shape=[224, 224, 3], name='VGG16'):

        super().__init__(shape, name=name)
        temp = self.ret_dict
        temp = self.nest(conv(temp['shape'], 64))
        temp = self.nest(conv(temp['shape'], temp['shape'][2]))
        temp = self.nest(pool(temp['shape']))

        temp = self.nest(conv(temp['shape'], temp['shape'][2]*2))
        temp = self.nest(conv(temp['shape'], temp['shape'][2]))
        temp = self.nest(pool(temp['shape']))

        temp = self.nest(conv(temp['shape'], temp['shape'][2]*2))
        temp = self.nest(conv(temp['shape'], temp['shape'][2]))
        temp = self.nest(conv(temp['shape'], temp['shape'][2]))
        temp = self.nest(pool(temp['shape']))

        temp = self.nest(conv(temp['shape'], temp['shape'][2]*2))
        temp = self.nest(conv(temp['shape'], temp['shape'][2]))
        temp = self.nest(conv(temp['shape'], temp['shape'][2]))
        temp = self.nest(pool(temp['shape']))

        temp = self.nest(conv(temp['shape'], temp['shape'][2]))
        temp = self.nest(conv(temp['shape'], temp['shape'][2]))
        temp = self.nest(conv(temp['shape'], temp['shape'][2]))
        temp = self.nest(pool(temp['shape']))

        temp = self.nest(flat(temp['shape']))
        temp = self.nest(fc(temp['shape'][2], 4096))
        temp = self.nest(fc(temp['shape'][2], temp['shape'][2]))
        temp = self.nest(fc(temp['shape'][2], 1000))
        

if __name__ == '__main__':

    # calc
    # ignore computations of activation functions, pooolings, etc.
    print('-'*32)
    print_madd(AlexNet().return_dict())

    print('-'*32)
    print_madd(VGG16().return_dict())
