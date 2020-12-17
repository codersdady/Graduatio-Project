"""=======================构建列表类============================"""

"""自定义列表Node，双向列表 + 头尾节点"""
"""
    data：   为列表节点存储的数据
    before： 为列表节点的上一个节点
    next：   为列表节点的下一个节点
"""
class ListNode(object):
    def __init__(self, data, before=None, next=None):
        self.data = data
        self.before = before
        self.next = next
        self.index = []

    def setData(self, data):
        self.data = data

    def setBefore(self, before):
        self.before = before

    def setNext(self, next):
        self.next = next



class BuildList():
    head = ListNode(None)   #列表头节点，不存储数据
    tail = ListNode(None, head, head)   #列表尾结点，不存储数据
    head.setBefore(tail)
    head.setNext(tail)
    size = 0    #列表节点个数，初始为0
    "构建列表的主方法"
    def build(self,data):
        for i, d in enumerate(data):
            if self.size == 0:
                "如果列表大小为0，新建一个Node节点"
                n = ListNode([], self.head, self.tail)
                n.data.append(d)
                n.index.append(i)
                self.head.next = n
                self.tail.before = n
                self.size += 1
            else:
                "列表不为空，比较数据大小，插入合适位置"
                node = self.head.next
                while node != self.tail:
                    if self.cmpNode(d, node) == -1:
                        node = node.before
                        break
                    elif self.cmpNode(d, node) == 1:
                        node = node.next
                    else:
                        break
                self.insertData(d, node, i)
        return self.head.next


    "插入数据 1.直接加入列表节点 2.构建新Node"
    def insertData(self, data, node, index):
        if node != self.head and node != self.tail:
            if self.cmpNode(data, node) == 1:
                n = ListNode([])
                n.before = node
                n.next = node.next
                n.before.next = n
                n.next.before = n
                n.data.append(data)
                n.index.append(index)
                self.size += 1
            else:
                node.data.append(data)
                node.index.append(index)
        else:
            n = ListNode([])
            n.data.append(data)
            n.index.append(index)
            self.size += 1
            if node == self.head:
                n.next = self.head.next
                n.before = self.head;
                n.next.before = n
                n.before.next = n
            else:
                n.before = self.tail.before
                n.next = self.tail
                n.before.next = n
                n.next.before = n

    "比较数据与Node节点中的数据集的大小，小于返回-1，等于返回0，大于返回1"
    def cmpNode(self,data,node):
        l = 0
        s = 0
        for i in node.data:
            if self.compare(data,i) == -1:
                s += 1
            elif self.compare(data,i) == 1:
                l += 1
        if l > s:
            return 1
        elif l < s:
            return -1
        else:
            return 0

    "比较两个数据值的大小，绝对小返回-1，绝对大返回1，不可比返回0"
    def compare(self,data1,data2):
        l = True
        s = True
        for i in range(len(data1)):
            if data1[i] > data2[i]:
                s = False
            elif data1[i] < data2[i]:
                l = False
        if (l and s) or (not l and not s):
            return 0
        elif l:
            return 1
        else:
            return -1


