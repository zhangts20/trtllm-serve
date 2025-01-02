#define CPPHTTPLIB_OPENSSL_SUPPORT
#include "httplib.h"
#include "session.hpp"

int main(int argc, char **argv, char **envp)
{
    httplib::Server server;

    server.Get("/hi", [](const httplib::Request &req, httplib::Response &res) {
        res.set_content("Hello World!\n", "text/plain");
    });

    LOG_INFO("Server available at localhost:8000");
    server.listen("localhost", 8000);
}