//
// Created by sonwi on 2022/9/11.
//

#ifndef RELEETCODE_SOLUTIONS_H
#define RELEETCODE_SOLUTIONS_H

#include <vector>
#include <bits/stdc++.h>
#include "DataStructs.h"
using namespace std;

template <typename T>
void printVector(vector<T>& nums) {
    cout << "[ ";
    for(T num : nums) {
        cout << num << " ";
    }
    cout <<"]\n";
}

template<typename T>
void printMatrix(vector<vector<T>> matrix) {
   for(auto& item : matrix) {
       printVector(item);
   }
}

// 代码随想录 数组篇
class ArraySolution {
public:
    //代码随想录 数组篇
    vector<int> sortedSquares(vector<int>& A) {
        vector<int> res;
        res.resize(A.size());

        int resIdx = A.size() - 1;
        int begin = 0, end = A.size() - 1;

        while(begin <= end) {
            int leftItem = A[begin] < 0 ? -A[begin] : A[begin];
            int rightItem = A[end] < 0 ? -A[end] : A[end];

            if(leftItem > rightItem) {
                res[resIdx--] = (leftItem * leftItem);
                begin++;
            } else {
                res[resIdx--] = (rightItem * rightItem);
                end--;
            }
        }

        return res;
    }

    void testSortedSquares() {
        vector<int> nums {-10, 2, 5, 8, 9, 10};
        vector<int> res = this->sortedSquares(nums);
        printVector(res);
    }

    int minSubArrayLen(int s, vector<int>& nums) {
        int left = 0, right = 1;
        int res = INT_MAX, sum = 0;

        bool plusLeft = false;
        while(right <= nums.size() && left < right) {
            if(!plusLeft) {
                sum += nums[right - 1];
            } else {
                sum -= nums[left - 1];
            }
            if(sum >= s) {
                res = min(res, right - left);
                left++;
                plusLeft = true;
            } else {
                if(right == nums.size()) {
                    return res;
                }
                right++;
                plusLeft = false;
            }
        }
        return res == INT_MAX ? 0 : res;
    }

    int minSubArrayLenV2(int target, vector<int>& nums) {
        int left = 0, sum = 0;
        int res = INT_MAX;

        for(int right = 0; right < nums.size(); ++right) {
            sum += nums[right];
            while(sum >= target) {
                res = min(res, right - left + 1);
                sum -= nums[left++];
            }
        }

        return res == INT_MAX ? 0 : res;
    }

    void testMinSubArrayLen() {
        int target = 7;
        vector<int> nums {2,3,1,2,4,3};

        cout << this->minSubArrayLenV2(target, nums) << endl;
        cout << INT_MAX << endl;
    }

    vector<vector<int>> generateMatrix(int n) {
        vector<vector<int>> res(n, vector<int>(n, 0));
        int loop = n / 2;
        int count = 1;
        int offset = 1;
        int startX = 0, startY = 0;

        while(loop--) {
            int i = startX, j = startY;

            for(; j < n - offset; ++j) {
                res[i][j] = count++;
            }

            for(; i < n - offset; ++i) {
                res[i][j] = count++;
            }

            for(; j > startX ; --j) {
                res[i][j] = count++;
            }

            for(; i > startX; --i ) {
                res[i][j] = count++;
            }

            offset++;
            startX++;startY++;
        }

        if(n % 2 == 1) {
            res[n/2][n/2] = count;
        }

        return res;
    }

    void testMatrix() {
        printMatrix(generateMatrix(4));
    }

};

// 代码随想录 链表篇
struct ListSolution {
    // 移除链表特定元素
    ListNode* removeElements(ListNode* head, int val) {
        ListNode node(0, head);
        ListNode *dummyNode = &node;

        ListNode *slow = dummyNode, *fast = head;
        while(fast) {
            if(fast->val == val) {
                fast = fast->next;
                slow->next = fast;
            } else {
                slow = fast;
                fast = fast->next;
            }
        }
        return dummyNode->next;
    }

    ListNode* reverseList(ListNode* head) {
        ListNode *pre = nullptr;
        ListNode *cur = head;

        while(cur) {
            ListNode *next = cur->next;
            cur->next = pre;
            pre = cur;
            cur = next;
        }

        return pre;
    }

    ListNode* recursiveReverse(ListNode* head) {
        ListNode *node = recursiveRe(head);
        if(!node) return node;
        ListNode *ret = node->next;
        node->next = nullptr;
        return ret;
    }

    ListNode* recursiveRe(ListNode* head) {
        if(head == nullptr) return nullptr;

        ListNode* node = recursiveRe(head->next);
        if(node == nullptr) {
            head->next = head;
            return head;
        }

        ListNode* tmp = node->next;
        node->next = head;
        head->next = tmp;
        return head;
    }

    void testRecursiveReverse() {
        ListNode *list = ListNode::makeList({1,2,3,4,5,6});
        list->print();
        recursiveReverse(list)->print();
    }

    // 交换相邻的两个节点
    ListNode* swapPairs(ListNode* head) {
        if(!head || !head->next) return head;

        ListNode dummyHead(0);
        ListNode *pre = &dummyHead;
        pre->next = head;
        ListNode* cur = head;
        ListNode* next = head->next;

        while(cur && cur->next) {
            ListNode* nnext = next->next;
            cur->next = next->next;
            next->next = cur;
            pre->next = next;

            pre = cur;
            cur = cur->next;
            if(nnext)next = nnext->next;
        }

        return dummyHead.next;
    }

    void testSwapPairs() {
        ListNode *list = ListNode::makeList({1,2,3,4});
        list->print();
        swapPairs(list)->print();
    }

    //删除链表的倒数第n个节点
    ListNode* removeNthFromEnd(ListNode* head, int n) {
        ListNode *dummyHead = new ListNode(0);
        dummyHead->next = head;
        ListNode *slow = dummyHead, *fast = dummyHead;

        while(n != 0 && fast->next != nullptr) {
            fast = fast->next;
            --n;
        }

        if(n != 0) return head;

        while(fast->next != nullptr) {
            slow = slow->next;
            fast = fast->next;
        }
        slow->next = slow->next->next;
        return dummyHead->next;
    }

    void testRemoveNthFromEnd() {
        ListNode *list = ListNode::makeList({1,2,3,4,5,6});
        list->print();
        removeNthFromEnd(list, 6)->print();
    }

    // 找到链表相交的入口
    ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
        ListNode *node1 = headA, *node2 = headB;
        while(node1 || node2) {
            if(node1 == node2) return node1;
            if(node1 == nullptr) {
                node1 = headB;
            }
            if(node2 == nullptr) {
                node2 = headA;
            }
        }
        return nullptr;
    }

    // 找到链表环的入口
    ListNode *detectCycle(ListNode *head) {
        ListNode *slow = head, *fast = head;

        while(fast && fast->next) {
            slow = slow->next;
            fast = fast->next->next;
            if(slow == fast) break;
        }

        if(slow != fast) return nullptr;

        slow = head;
        while(slow != fast) {
            slow = slow->next;
            fast = fast->next;
        }

        return slow;
    }
};

struct HashSolution {

};

struct BinaryTreeSolution {
    // 翻转二叉树
    TreeNode* invertTree(TreeNode* root) {
        if(root == nullptr) return root;
        swap(root->left, root->right);
        invertTree(root->left);
        invertTree(root->right);
        return root;
    }

    // 检查二叉树是否对称
    bool isSymmetric(TreeNode* root) {
        if(!root) return false;
        return help(root->left, root->right);
    }

    bool help(TreeNode* node1, TreeNode* node2) {
        if(!node1 && !node2) return true;
        if(!node1 || !node2) return false;
        if(node1->val != node2->val) return false;
        return help(node1->left, node2->right) && help(node1->right, node2->left);
    }

    // 求二叉树的最大深度
    int maxDepth(TreeNode* root) {
        if(root == nullptr) return 0;
        return 1 + max(maxDepth(root->left), maxDepth(root->right));
    }

};

//707 设计链表
class MyLinkedList {
public:
    ListNode *dummyHead;
    ListNode *dummyEnd;
    int numOfItem;
public:
    MyLinkedList() {
        dummyHead = new ListNode(0);
        dummyEnd = new ListNode(0);
        dummyHead->next = dummyEnd;
        numOfItem = 0;
    }

    int get(int index) {
        ListNode *cur = dummyHead;
        while(cur->next != dummyEnd && index-- != 0) {
            cur = cur->next;
        }
        if(cur->next == dummyEnd) return -1;
        if(index == -1) return cur->next->val;
        return -1;
    }

    void addAtHead(int val) {
        ListNode *node = new ListNode(val);
        node->next = dummyHead->next;
        dummyHead->next = node;
        numOfItem++;
    }

    void addAtTail(int val) {
        addAtIndex(numOfItem, val);
    }

    void addAtIndex(int index, int val) {
        if(index < 0 ) index = 0;
        if(index >= numOfItem) index = numOfItem;

        ListNode *slow = dummyHead;
        ListNode *fast = dummyHead->next;

        while(index--) {
            slow = fast;
            fast = fast->next;
        }

        slow->next = new ListNode(val);
        slow->next->next = fast;
        numOfItem++;
    }

    void deleteAtIndex(int index) {
        if(index < 0 || index >= numOfItem) return;

        ListNode *slow = dummyHead;
        ListNode *fast = dummyHead->next;

        while(index--) {
            slow = fast;
            fast = fast->next;
        }

        slow->next = fast->next;
        delete fast;
        numOfItem--;
    }
};

#endif //RELEETCODE_SOLUTIONS_H
