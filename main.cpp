#include <iostream>
#include <thread>
#include <vector>
#include <torchvision/models/resnet.h>


int main()
{
    auto model = vision::models::ResNet18();
    model->eval();
    model->to(torch::kCUDA);
    auto worker_thread = [&model](){
        std::cout << "thread " << std::this_thread::get_id() << " start" << std::endl;
        for (int i = 0;i < 10000; ++i) {
            auto in = torch::rand({1, 3, 10, 10});
            auto gpu_in = in.to(torch::kCUDA);
            auto gpu_out = model->forward(gpu_in);
            auto cpu_out = gpu_out.to(torch::kCPU);
        }
        std::cout << "thread " << std::this_thread::get_id() << " ends" << std::endl;
    };

    auto core_num = std::thread::hardware_concurrency();
    std::cout << "starting test with " << core_num << " threads" << std::endl;
    std::vector<std::thread> worker_threads;

    for (uint32_t i = 0;i < core_num; ++i) {
        worker_threads.emplace_back(worker_thread);
    }

    for (uint32_t i = 0;i < core_num; ++i) {
        worker_threads[i].join();
    }

    std::cout << "test success" << std::endl;
    return 0;
}