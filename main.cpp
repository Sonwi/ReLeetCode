#include <iostream>

#include "Solutions.h"
#include "Interview.h"
#include "exam.h"
#include <string>

int main() {
    //test entry

    // 数组篇
//    ArraySolution s1;
//    s1.testMatrix();

    // 链表篇

//    MyLinkedList linkedList;
//    linkedList.addAtHead(1);
//    linkedList.addAtTail(3);
//    linkedList.addAtIndex(1,2);   //链表变为1-> 2-> 3
//    cout << linkedList.get(1);            //返回2
//    cout << endl;
//    linkedList.deleteAtIndex(1);  //现在链表是1-> 3
//    cout << linkedList.get(1);            //返回3
//    cout << endl;
//
//    linkedList.dummyHead->next->print();
//    ListSolution s2;
//    s2.testRemoveNthFromEnd();

    // for interview
//    Interview item;

    // 二叉树篇
//    BinaryTreeSolution *s3;
//    TreeNode *tree = TreeNode::makeBinaryTree({1,2,2,null,5,5,4});
//    tree->print();
//    cout << "-----------" << endl;
//    cout << s3->isSymmetric(tree) << endl;

    // hash表篇
    HashSolution h;
//    cout << h.isAnagram("awert", "werta") << endl;

    // DP
    DynamicProgramming dp;
//    vector<int> nums{10, 15, 20};
//    dp.minCostClimbingStairs(nums);
//    dp.uniquePathsWithObstacles({{0,0,0},{0,1,0},{0,0,0}});
//    dp.testIntegerBreak();
//    dp.testNumOfSeqs();

    BigData b;
//    b.testAdd();
//    b.testMultipy();
    //test entry end
    // comment from manjaro
    Sort s;
    s.testSort();
    return 0;
}
