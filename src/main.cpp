#include <iostream>
#include <cstdlib>

#include "core/application.h"

int main()
{
    Application app;

    try
    {
        app.Run();
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << "\n";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

