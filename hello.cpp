#include <iostream>
#include <unistd.h>

int main(int argc, char* argv[]) {
    char hostname[128];
    if (gethostname(hostname, sizeof(hostname)) != 0) {
        std::cerr << "Error getting hostname" << std::endl;
        return 1;
    }
    std::cout << "New hello from " << hostname << std::endl;

    return 0;
}
