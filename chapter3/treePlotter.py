import matplotlib.pyplot as plt

# 定义文本框和箭头格式
decision_node = dict(boxstyle='sawtooth', fc='0.8')
leaf_node = dict(boxstyle='round4', fc='0.8')
arrow_args = dict(arrowstyle='<-')


def get_num_leafs(my_tree):
    num_leafs = 0
    firstStr = my_tree.keys()[0]
    secondDict = my_tree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[
                    key]).__name__ == 'dict':  # test to see if the nodes are dictonaires, if not they are leaf nodes
            num_leafs += get_num_leafs(secondDict[key])
        else:
            num_leafs += 1
    return num_leafs


def get_tree_depth(my_tree):
    maxDepth = 0
    firstStr = my_tree.keys()[0]
    secondDict = my_tree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[
                    key]).__name__ == 'dict':  # test to see if the nodes are dictonaires, if not they are leaf nodes
            thisDepth = 1 + get_tree_depth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth: maxDepth = thisDepth
    return maxDepth


def plot_node(nodeTxt, centerPt, parentPt, nodeType):
    create_plot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)


def plot_mid_text(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    create_plot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)


def plot_tree(my_tree, parentPt, nodeTxt):  # if the first key tells you what feat was split on
    num_leafs = get_num_leafs(my_tree)  # this determines the x width of this tree
    depth = get_tree_depth(my_tree)
    firstStr = my_tree.keys()[0]  # the text label for this node should be this
    cntrPt = (plot_tree.xOff + (1.0 + float(num_leafs)) / 2.0 / plot_tree.totalW, plot_tree.yOff)
    plot_mid_text(cntrPt, parentPt, nodeTxt)
    plot_node(firstStr, cntrPt, parentPt, decision_node)
    secondDict = my_tree[firstStr]
    plot_tree.yOff = plot_tree.yOff - 1.0 / plot_tree.totalD
    for key in secondDict.keys():
        if type(secondDict[
                    key]).__name__ == 'dict':  # test to see if the nodes are dictonaires, if not they are leaf nodes   
            plot_tree(secondDict[key], cntrPt, str(key))  # recursion
        else:  # it's a leaf node print the leaf node
            plot_tree.xOff = plot_tree.xOff + 1.0 / plot_tree.totalW
            plot_node(secondDict[key], (plot_tree.xOff, plot_tree.yOff), cntrPt, leaf_node)
            plot_mid_text((plot_tree.xOff, plot_tree.yOff), cntrPt, str(key))
    plot_tree.yOff = plot_tree.yOff + 1.0 / plot_tree.totalD


# if you do get a dictonary you know it's a tree, and the first element will be another dict

def create_plot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    create_plot.ax1 = plt.subplot(111, frameon=False, **axprops)  # no ticks
    # create_plot.ax1 = plt.subplot(111, frameon=False) #ticks for demo puropses 
    plot_tree.totalW = float(get_num_leafs(inTree))
    plot_tree.totalD = float(get_tree_depth(inTree))
    plot_tree.xOff = -0.5 / plot_tree.totalW
    plot_tree.yOff = 1.0
    plot_tree(inTree, (0.5, 1.0), '')
    plt.show()


# def create_plot():
#    fig = plt.figure(1, facecolor='white')
#    fig.clf()
#    create_plot.ax1 = plt.subplot(111, frameon=False) #ticks for demo puropses 
#    plot_node('a decision node', (0.5, 0.1), (0.1, 0.5), decision_node)
#    plot_node('a leaf node', (0.8, 0.1), (0.3, 0.8), leaf_node)
#    plt.show()

def retrieve_tree(i):
    listOfTrees = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                   {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                   ]
    return listOfTrees[i]

    # create_plot(thisTree)

if __name__ == '__main__':
    my_t = retrieve_tree(0)
    print(my_t)
    create_plot(my_t)
