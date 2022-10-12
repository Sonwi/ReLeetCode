//
// Created by sonwi on 2022/9/11.
//

#ifndef RELEETCODE_SOLUTIONS_H
#define RELEETCODE_SOLUTIONS_H

#include <vector>
#include <bits/stdc++.h>
#include "DataStructs.h"
using namespace std;

namespace alex{
    template <typename T>
    void printVector(vector<T> nums) {
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

    vector<int> randNums(int size) {
        vector<int> res; res.reserve(size);

        srand(time(0));
        for(int i = 0; i < size; ++i) {
            res.push_back(rand() % 1000);
        }
        return res;
    }
}

// 代码随想录 数组篇
struct ArraySolution {
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
        alex::printVector(res);
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
        alex::printMatrix(generateMatrix(4));
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
    // 242 有效的字母异位分词
    bool isAnagram(string s, string t) {
        if(s.size() != t.size()) return false;
        vector<int> arraySet(26, 0);
        for(char c : s) {
            arraySet[c - 'a'] ++;
        }

        for(char c : t) {
            if(--arraySet[c - 'a'] < 0) {
                return false;
            }
        }
        return true;
    }

    // 349 两个数组的交集
    vector<int> intersection(vector<int>& nums1, vector<int>& nums2) {
        vector<int> arraySet(1001, 0);
        vector<int> res;
        for(int num : nums1) {
            arraySet[num]++;
        }

        for(int num : nums2) {
            if(arraySet[num] > 0) {
                arraySet[num] = -1;
                res.push_back(num);
            }
        }

        return res;
    }

    // 202 快乐数
    bool isHappy(int n) {
        if(n < 10) return n == 1 || n == 7;
        int sum = 0;
        while(n) {
            sum += (n%10) * (n%10);
            n = n/10;
        }
        return isHappy(sum);
    }
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

    // 二叉搜索树最近公共祖先
    // 充分利用二叉搜索树的特性
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode *q) {
        if(p->val > q->val) swap(p, q);
        TreeNode *cur = root;
        while(cur) {
            if(cur->val >= p->val && cur->val <= q->val) {
                return cur;
            } else if(cur->val < p->val) {
                cur = cur->right;
            } else {
                cur = cur->left;
            }
        }
        return cur;
    }
};

struct DynamicProgramming {
    // 爬楼梯
    int climbStairs(int n) {
        if(n == 0 || n == 1) return 1;
        int dp[2];
        dp[0] = 1, dp[1] = 1;
        int res = 0;
        for(int i = 2; i <= n; ++i) {
            res = dp[0] + dp[1];
            dp[0] = dp[1];
            dp[1] = res;
        }
        return res;
    }
    // 最小花费爬楼梯
    int minCostClimbingStairs(vector<int>& cost) {
        if(cost.size() <= 1) {
            return 0;
        }
        vector<int> price(2, 0);
        for(int i = 2; i <= cost.size(); ++i) {
            int tmp = min(price[1] + cost[i-1], price[0] + cost[i-2]);
            price[0] = price[1];
            price[1] = tmp;
        }
        return price.back();
    }

    //62 不同路径
    int uniquePaths(int m, int n) {
        vector<vector<int>> board(m, vector<int>(n, 0));
        // 边缘节点路径初始化
        for(int i = 0; i < n; ++i) {
            board[0][i] = 1;
        }
        for(int i = 0; i < m; ++i) {
            board[i][0] = 1;
        }
        for(int i = 1; i < m; ++i) {
            for(int j = 1; j < n; ++j ) {
                board[i][j] = board[i-1][j] + board[i][j-1];
            }
        }
        return board.back().back();
    }

    //63 带有障碍物的不同路径
    int uniquePathsWithObstacles(vector<vector<int>> obstacleGrid) {
        int m = obstacleGrid.size();
        if(m == 0) return 0;
        int n = obstacleGrid[0].size();
        vector<vector<int>> dp(m, vector<int>(n, 0));
        // 初始化边界
        for(int i = 0; i < n; ++i) {
            if(obstacleGrid[0][i] == 1) break;
            dp[0][i] = 1;
        }
        for(int i = 0; i < m; ++i) {
            if(obstacleGrid[i][0] == 1) break;
            dp[i][0] = 1;
        }
        for(int i = 1; i < m; ++i) {
            for(int j = 1; j < n; ++j) {
                dp[i][j] = (obstacleGrid[i-1][j] == 1 ? 0 : dp[i-1][j]) + (obstacleGrid[i][j-1] == 1? 0 : dp[i][j-1]);
            }
        }
        return dp[m-1][n-1];
    }

    // 343 整数拆分
    int integerBreak(int n) {
        vector<int>nums(n + 1, 0);
        nums[1] = 1, nums[2] = 1;
        for(int i = 3; i < n + 1; ++i) {
            int maxMulti = 0;
            for(int j = 1; j < i; ++j){
                maxMulti = max(maxMulti, max((i - j) * j, (i - j) * nums[j]));
            }
            nums[i] = maxMulti;
        }
        return nums[n];
    }
    void testIntegerBreak() {
        integerBreak(10);
    }


    // 96 不同的二叉搜索树
    int numTrees(int n) {
        vector<int> nums(n + 1, 0);
        nums[0] = 1;
        nums[1] = 1;
        for (int i = 2; i < n + 1; ++i) {
            int numOfTree = 0;
            for (int j = 1; j <= i; ++j) {
                numOfTree += nums[j - 1] * nums[i - j];
            }
            nums[i] = numOfTree;
        }
        return nums[n];
    }

        // 美团笔试题: 只有 1 和 -1 数组，求积为1的序列个数
    int numOfSeqs(vector<int> nums) {
        vector<vector<int>> dp(2, vector<int>(nums.size(), 0));
        dp[0][0] = nums[0] == 1 ? 1 : 0;
        dp[0][1] = nums[0] == -1 ? 1 : 0;
        int ret = dp[0][0];
        for(int i = 1; i < nums.size(); ++i) {
            if(nums[i] == 1) {
                dp[0][i] = dp[0][i - 1] + 1;
                dp[1][i] = dp[1][i-1];
            }else {
                dp[0][i] = dp[1][i-1];
                dp[1][i] = dp[0][i-1] + 1;
            }
            ret += dp[0][i];
        }
        return ret;
    }

    void testNumOfSeqs() {
        cout << numOfSeqs({1,1,-1,-1});
    }
};

struct BigData {
    // 大数相乘
    //1. num1[i] * num2[j] 贡献给特定位置 i + j + 1 (进位在 i+j)
    //2. 相乘 最多 num1.size() + num2.size() 位数
    string multiply(string num1, string num2) {
        vector<int> res(num1.size() + num2.size());

        for(int i = 0; i < num1.size(); ++i) {
            for(int j = 0; j < num2.size(); ++j) {
                res[i + j + 1] += (num1[i] - '0') * (num2[j] - '0');
            }
        }

        for(int i = res.size() - 1; i >= 1; --i) {
            res[i-1] += res[i]/10;
            res[i] = res[i] % 10;
        }

        int idx = res[0] == 0? 1 : 0;
        string resStr; resStr.reserve(res.size());
        for(; idx < res.size(); ++idx) {
            resStr.push_back(res[idx] + '0');
        }
        return resStr;
    }

    void testMultipy() {
        cout << 99 * 98 << endl;
        cout << multiply("99", "98");
    }

    // 大数相加
    string add(string s, string t) {
        if(s.size() < t.size())
            swap(s,t);
        string ret;
        ret.reserve(s.size() + 1);

        int sum = 0;
        int up = 0;
        int i = s.size() - 1, j = t.size() - 1;

        while(i >= 0 || j >= 0 || up != 0) {
            int a = i >= 0? s[i] - '0' : 0;
            int b = j >= 0? t[j] - '0' : 0;
            sum  = a + b + up;
            up  = sum / 10;
            sum = sum % 10;
            ret.push_back(sum + '0');
            --i;--j;
        }
        reverse(ret.begin(), ret.end());
        return ret;
    }

    void testAdd() {
        cout << 99 + 108 << endl;
        cout << add("99", "108");
    }
};

struct CodeTop {
    // 3. 无重复字符的最长子串
    int lengthOfLongestSubstring(string s) {
        if(s.size() <= 1) return s.size();

        unordered_map<char, int>myMap;
        int res = 1;
        int left = 0, right = 0;
        for(; right < s.size(); ++right) {
            if(myMap.count(s[right]) && myMap[s[right]] >= left) {
                left = myMap[s[right]] + 1;
            }

            myMap[s[right]] = right;
            res = max(res, right - left + 1);
        }
        return res;
    }

    void testlengthOfLongestSubstring() {
        cout << lengthOfLongestSubstring("abba") << endl;
    }

    //O(n) 复杂度找到无序数组第k大的元素
    int findKthLargest(vector<int>& nums, int k) {
        buildHead(nums, nums.size());
        for(int i =0 ; i < k; ++i) {
            swap(nums[0], nums[nums.size()-1-i]);
            sink(nums, 0, nums.size()-1-i);
        }
        return nums[nums.size()-k];
    }

    void buildHead(vector<int>& nums, int n) {
        int k = (n-2)/2;
        while(k >= 0) {
            sink(nums, k, n);
            --k;
        }
    }

    void sink(vector<int>& nums, int k, int n) {
        while(2*k + 1 < n) {
            int j = 2*k+1;
            if(j+1 < n && nums[j+1] > nums[j]) ++j;
            if(nums[k] >= nums[j]) break;
            swap(nums[k], nums[j]);
            k = j;
        }
    }

    // 33. 搜索旋转排序数组
    int search(vector<int>& nums, int target) {
        int left = 0, right = nums.size() - 1;
        while(left <= right) {
            int mid = (left + right)/2;
            if(nums[mid] == target) return mid;
            if(nums[mid] >= nums[0]) {
                if(nums[mid] > target && target >= nums[0]) right = mid -1;
                else left = mid + 1;
            } else {
                if(nums[mid] < target && target <= nums.back()) left = mid + 1;
                else right = mid - 1;
            }
        }
        return -1;
    }

    void testSearch() {
        vector<int> nums{4,5,6,7,0,1,2};
         cout << search(nums, 0) << endl;
    }

    //200 岛屿数量
    int numIslands(vector<vector<char>>& grid) {
        int n = grid.size();
        if(n == 0) return 0;
        int m = grid[0].size();
        vector<vector<int>> visited(n, vector<int>(m, 0));
        int res = 0;
        for(int i = 0; i < n; ++i) {
            for(int j = 0; j < m; ++j) {
                if(visited[i][j] == 0 && grid[i][j] == '1') {
                    ++res;
                    depFirst(visited, grid, i, j);
                }
            }
        }
        return res;
    }

    void depFirst(vector<vector<int>>& visited, vector<vector<char>>& grid, int i, int j) {
        if(grid[i][j] == '0') return;
        visited[i][j] = 1;
        if(i-1 >= 0)depFirst(visited, grid, i-1, j);
        if(i+1 < visited.size())depFirst(visited, grid, i+1, j);
        if(j-1 >= 0)depFirst(visited, grid, i, j-1);
        if(j+1 < visited[0].size())depFirst(visited, grid, i, j+1);
    }

    //5 最长回文子串
    // 注意使用 dp[i][j] 表示 i - j的子串
    string longestPalindrome(string s) {
        vector<vector<int>> dp(s.size(), vector<int>(s.size()));
        for(int i = 0 ; i < s.size(); ++i) {
            dp[i][i] = 1;
        }

        int begin = 0, maxLen = 0;
        for(int len = 2; len < s.size() + 1; ++len) {
            for(int i = 0; i < len;++i) {
                int right = i + len - 1;
                if(right >= s.size()) break;
                if(s[i] != s[right]) dp[i][right] = 0;
                else {
                    if(right - i < 3) {
                        dp[i][right] = 1;
                    } else {
                        dp[i][right] = dp[i+1][right-1];
                    }
                }
                if(dp[i][right] == 1 && right - i + 1 > maxLen) {
                    maxLen = right - i + 1;
                    begin = i;
                }
            }
        }
        return s.substr(begin, maxLen);
    }

    void testLong(){
        longestPalindrome("cbbd");
    }

    // 236 二叉树的最近公共祖先
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        if(root == p || root == q || ! root) return root;
        TreeNode* left = lowestCommonAncestor(root->left, p, q);
        TreeNode* right = lowestCommonAncestor(root->right, p, q);
        if(left && right) return root;
        else if(!left) return right;
        else return left;
    }

    // 求平方根
    int sqrt(int x) {
        // write code here
        if(x == 0 || x == 1) return x;
        int left = 1, right = x;
        while(left <= right) {
            int mid = (left + right) / 2;
            if(mid <= x/mid && (mid+1) >x/(mid+1)) return mid;
            if(mid > x/mid) right = mid - 1;
            else left = mid + 1;
        }
        return 0;
    }

};

struct NowCoder {
    
    // 翻转 m-n 之间的链表
    ListNode* reverseBetween(ListNode* head, int m, int n) {
        // write code here
        ListNode* dummy = new ListNode(0);
        dummy->next = head;
        
        ListNode *lo = head;
        ListNode *pre = dummy;
        for(int i = 0; i < m-1; ++i) {
            lo = lo->next;
            pre = pre->next;
        }
        
        ListNode *seqPre = pre;
        ListNode *next = lo->next;
        for(int i = 0; i < n - m + 1; ++i) {
            lo->next = pre;
            pre = lo;
            lo = next;
            next = lo->next;
        }
        
        seqPre->next->next = lo;
        seqPre->next = pre;
        return dummy->next;
    }
    
    // k个一组翻转链表
    ListNode* reverseKGroup(ListNode* head, int k) {
        // write code here
        int numOfNodes = 0;
        ListNode* cur = head;
        while(cur) {
            ++numOfNodes;
            cur = cur->next;
        }
        
        ListNode* dummy = new ListNode(0);
        dummy->next = head;
        
        ListNode* seqPre = dummy;
        ListNode* pre = dummy;
        cur = head;
        for(int i = 0; i < numOfNodes/k; ++i) { // 组数量
            for(int j = 0; j < k; ++j) {
                ListNode* next = cur->next;
                cur->next = pre;
                pre = cur;
                cur = next;
            }
            seqPre->next->next = cur;
            seqPre->next = pre;
            seqPre = pre;
        }
        return dummy->next;
    }
    
    void testReverseKGroup() {
        ListNode* list = ListNode::makeList({1,2,3,4,5});
        reverseKGroup(list, 2);
    }
    
    // BM4 合并两个排序的链表
    ListNode* Merge(ListNode* pHead1, ListNode* pHead2) {
        ListNode* dummy = new ListNode(0);
        ListNode* cur = dummy;
        
        ListNode* l1 = pHead1;
        ListNode* l2 = pHead2;
        
        while(l1 || l2) {
            if(!l1) {
                cur->next = l2;
                l2 = l2->next;
            }else if(!l2) {
                cur->next = l1;
                l1 = l1->next;
            }else if(l1->val <= l2->val) {
                cur->next = l1;
                l1 = l1->next;
            }else {
                cur->next = l2;
                l2 = l2->next;
            }
            cur = cur->next;
        }
        return dummy->next;
    }
    
    // BM5 合并k个已经排序的链表
    ListNode *mergeKLists(vector<ListNode *> &lists) {
        return mergeLists(lists, 0, lists.size() - 1);
    }
    
    ListNode *mergeLists(vector<ListNode *> &lists, int sIdx, int eIdx) {
        if(sIdx >= eIdx) return nullptr;
        
        int mid = (sIdx + eIdx)/2;
        
        ListNode* left = mergeLists(lists, sIdx, mid);
        ListNode* right = mergeLists(lists, mid+1, eIdx);
        
        if(!left)return right;
        else if(!right) return left;
        else {
            return Merge(left, right);
        }
    }
};

// 146 LRU 缓存
class LRUCache {
private:
    struct node {
        int key;
        int value;
        node* pre;
        node* next;
        node(int key, int value): key(key), value(value), pre(nullptr), next(nullptr){}
    };
    int cap;
    unordered_map<int, node*> myMap;
    node* dummyHead;
    node* dummyTail;
public:
    LRUCache(int capacity) {
        dummyHead = new node(0, 0);
        dummyTail = new node(0, 0);
        dummyHead->next = dummyTail;
        dummyTail->pre = dummyHead;
        cap = capacity;
    }

    int get(int key) {
        if(myMap.count(key)) {
            removeToHead(myMap[key]);
            return myMap[key]->value;
        }
        return -1;
    }

    void put(int key, int value) {
        if(myMap.count(key)) {
            myMap[key]->value = value;
            removeToHead(myMap[key]);
            return;
        }
        node* tmp = new node(key, value);
        if(cap == 0) {
            deleteNode(dummyTail->pre);
            cap++;
        }
        cap--;
        addToHead(tmp);
    }

    void deleteNode(node* n) {
        n->pre->next = n->next;
        n->next->pre = n->pre;
        myMap.erase(n->key);
    }
    void addToHead(node* n) {
        n->next = dummyHead->next;
        dummyHead->next->pre = n;
        n->pre = dummyHead;
        dummyHead->next = n;
        myMap[n->key] = n;
    }
    void removeToHead(node* n) {
        deleteNode(n);
        addToHead(n);
    }
};

struct Sort {
    void testSort() {
        vector<int> nums {2,4,1,5,123,56,78,12,9,5};
        alex::printVector(selectSort(alex::randNums(20)));
        alex::printVector(insertSort(nums));
        //mergeSort(nums, 0, nums.size()-1);
//        quickSort(nums, 0, nums.size() -1);
        heapSort(nums, nums.size());
        alex::printVector(nums);
    }
    // 选择排序
    // 缺点：n^2 比较复杂度
    // 优点: 线性交换次数
    vector<int> selectSort(vector<int> nums) {
        for(int i = 0; i < nums.size(); ++i) {
            int min = i;
            for(int j = i; j < nums.size(); ++j) {
                if(nums[j] < nums[min]) min = j;
            }
            swap(nums[i], nums[min]);
        }
        return nums;
    }

    // 选择排序 遍历**后面**的数选择最小的放入
    // 插入排序 将拿到的数遍历**前面**插入
    vector<int> insertSort(vector<int> nums) {
        for(int i = 0; i < nums.size(); ++i) {
            for(int j = i; j > 0 && nums[j] < nums[j - 1]; --j) {
                swap(nums[j], nums[j-1]);
            }
        }
        return nums;
    }

    void quickSort(vector<int>& nums, int lo, int hi) {
        if(lo >= hi) return;
        int mid = partition(nums, lo, hi);
        quickSort(nums, lo, mid - 1);
        quickSort(nums, mid+1, hi);
    }

    int partition(vector<int>& nums, int lo, int hi) {
        srand(time(0));
        int randIdx = rand() %(hi - lo + 1) + lo;
        swap(nums[lo], nums[randIdx]);

        int item = nums[lo];
        int i = lo, j = hi+1;
        while(1) {
            while(nums[++i] < item){
                if(i >= hi) break;
            }
            while(nums[--j] > item){
                if(j <= lo) break;
            }
            if(i >= j) break;
            swap(nums[i], nums[j]);
        }
        swap(nums[lo], nums[j]);
        return j;
    }

    void mergeSort(vector<int>& nums, int lo, int hi) {
        if(lo >= hi) return;
        int mid = lo + (hi-lo)/2;
        mergeSort(nums, lo, mid);
        mergeSort(nums, mid+1, hi);
        merge(nums, lo, mid, hi);
    }

    void merge(vector<int>& nums, int lo, int mid, int hi) {
        vector<int> aux(nums.begin()+lo, nums.begin()+hi + 1);
        int i = 0, j = mid-lo + 1;
        int idx = lo;
        while(i <= mid - lo || j <= hi - lo) {
            if (i > mid - lo) nums[idx++] = aux[j++];
            else if (j > hi - lo) nums[idx++] = aux[i++];
            else if (aux[i] < aux[j]) nums[idx++] = aux[i++];
            else nums[idx++] = aux[j++];
        }
    }

    void heapSort(vector<int>& nums, int n) {
        // 从最后一个有子节点的开始下沉，构造堆
        for(int k = (n-2)/2; k >=0; --k) {
            sink(nums, k, n);
        }

        // 下沉 sink 是指这个节点打乱了原有的堆结构，从顶向下下沉时不能保证下沉时是堆结构
//        for(int k = 0; k <= (n-2)/2; ++k) {
//            sink(nums, k, n);
//        }

        // 每次把堆最大元素移到末尾，减小堆大小
        while(n > 1) {
            swap(nums[0], nums[n-1]);
            --n;
            sink(nums, 0, n);
        }
    }

    void sink(vector<int>& nums, int k, int n/*堆大小*/) {
        while(2*k + 1 < n) {
            int j = 2 * k + 1;
            if(j < n -1  && nums[j] < nums[j+1]) ++j;
            if(nums[k] >= nums[j]) break;
            swap(nums[k], nums[j]);
            k = j;
        }
    }
};

// 并查集
struct UnionFind{
    vector<int> id;
    int count;
    UnionFind(int n) : id(n), count(n) {
        for(int i = 0; i < n; ++i) {
            id[i] = i;
        }
    }
    void union_item(int p, int q) {
        int pRoot = find(p);
        int qRoot = find(q);
        if (pRoot != qRoot)
        {
            count--;
            id[pRoot] = qRoot;
        }
    }

    int find(int p) {
        if(p == id[p]) return p;
        return id[p] = find(id[p]);
    }

    bool connected(int p, int q) {
        return find(p) == find(q);
    }

    void test() {
        union_item(1,2);
        union_item(2,3);
        union_item(3,4);
        union_item(4,5);
        union_item(5,6);
        union_item(6,7);
        connected(1,7);
        alex::printVector(id);
    }

};

// 生产者消费者

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
