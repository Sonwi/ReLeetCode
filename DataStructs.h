//
// Created by sonwi on 2022/9/12.
//

#ifndef RELEETCODE_DATASTRUCTS_H
#define RELEETCODE_DATASTRUCTS_H

#define null INT_MIN

#include <vector>
#include <bits/stdc++.h>
using namespace std;

// 单链表
struct ListNode {
    int val;  // 节点上存储的元素
    ListNode *next;  // 指向下一个节点的指针
    ListNode(int x) : val(x), next(NULL) {}  // 节点的构造函数

    ListNode(int x, ListNode* next) : val(x), next(next) {}

    static ListNode * makeList(vector<int>&& nums) {
        if(nums.size() == 0) return nullptr;

        ListNode *head = new ListNode(0);
        ListNode *cur = head;
        for(int num : nums) {
            cur->next = new ListNode(num);
            cur = cur->next;
        }

        return head->next;
    }

    void print() {
        ListNode *node = this;
        while(node) {
            cout << node->val;
            if(node->next) {
                cout << "->";
            }
            node = node->next;
        }
        cout << endl;
    }
};

struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode() : val(0), left(nullptr), right(nullptr) {}
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}

    static TreeNode* makeBinaryTree(vector<int>&& nums) {
        int len = nums.size();
        if(len == 0) return nullptr;
        TreeNode *root = new TreeNode(nums[0]);
        sub(root, nums, 0);
        return root;
    }

    static void sub(TreeNode* node, vector<int>& nums, int idx) {
        if(idx * 2 + 1 < nums.size()) {
            if(nums[idx*2 + 1] != null) {
                node->left = new TreeNode(nums[idx*2+1]);
                sub(node->left, nums, idx*2+1);
            }
        }
        if(idx * 2 + 2 < nums.size()) {
            if(nums[idx*2+2] != null) {
                node->right = new TreeNode(nums[idx * 2 + 2]);
                sub(node->right, nums, idx * 2 + 2);
            }
        }
    }

    void print() {
        queue<TreeNode*> myQue;
        myQue.push(this);
        while(!myQue.empty()) {
            int size = myQue.size();
            for(int i = 0; i < size; ++i) {
                if(checkQueue(myQue)) {
                    while(!myQue.empty()) {
                        myQue.pop();
                    }
                    break;
                }
                TreeNode *node = myQue.front(); myQue.pop();
                if(node == nullptr) {
                    cout << "null ";
                    continue;
                }
                cout << node->val << " ";
                myQue.push(node->left);
                myQue.push(node->right);
            }
            cout << endl;
        }
    }

    bool checkQueue(queue<TreeNode*> que) {
        while(!que.empty()) {
            TreeNode* node = que.front(); que.pop();
            if(node != nullptr) return false;
        }
        return true;
    }
};

#endif //RELEETCODE_DATASTRUCTS_H
