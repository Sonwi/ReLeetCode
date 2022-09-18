//
// Created by sonwi on 2022/9/14.
//

#ifndef RELEETCODE_INTERVIEW_H
#define RELEETCODE_INTERVIEW_H

#include <bits/stdc++.h>
#include "DataStructs.h"

struct Interview {
    int getMinCost(int n) {
        int cost = 0, milk = 0, head = 0;
        int price = 5;
        while(milk < n) {
            cost += price;
            ++milk;
            ++head;
            if(milk < n && head == 2) {
                cost++;
                milk++;
                head = 1;
            }
        }
        return cost;
    }

    int maxSub(vector<int> nums) {
        if(nums.size() < 2) return 0;
        sort(nums.begin(), nums.end());

        int ret = INT_MIN;
        for(int i = 1; i < nums.size(); ++i) {
            ret = max(ret, nums[i] - nums[i-1]);
        }
        return ret;
    }


};

#endif //RELEETCODE_INTERVIEW_H
