var element = document.getElementById("blog_nav_script");

if (element == null) {
    document.write(`
    <nav class="navbar navbar-expand-lg navbar-light fixed-top" id="mainNav">
        <div class="container">
            <a class="navbar-brand js-scroll-trigger" href="index.html">Home</a><button class="navbar-toggler navbar-toggler-right" type="button" data-toggle="collapse" data-target="#navbarResponsive" aria-controls="navbarResponsive" aria-expanded="false" aria-label="Toggle navigation">Menu<i class="fas fa-bars"></i></button>
            <div class="collapse navbar-collapse" id="navbarResponsive">
                <ul class="navbar-nav ml-auto">
                    <li class="nav-item"><a class="nav-link js-scroll-trigger" href="projects.html">Projects</a></li>
                    <li class="nav-item"><a class="nav-link js-scroll-trigger" href="resume.html">Resume</a></li>
                    <li class="nav-item"><a class="nav-link js-scroll-trigger" href="contact.html">Contact</a></li>
                    <li class="nav-item"><a class="nav-link js-scroll-trigger" href="blog.html">Blog</a></li>
                </ul>
            </div>
        </div>
    </nav>
    `)
} else {
    document.write(`
    <nav class="navbar navbar-expand-lg navbar-light fixed-top" id="mainNav">
        <div class="container">
            <a class="navbar-brand js-scroll-trigger" href="../index.html">Home</a><button class="navbar-toggler navbar-toggler-right" type="button" data-toggle="collapse" data-target="#navbarResponsive" aria-controls="navbarResponsive" aria-expanded="false" aria-label="Toggle navigation">Menu<i class="fas fa-bars"></i></button>
            <div class="collapse navbar-collapse" id="navbarResponsive">
                <ul class="navbar-nav ml-auto">
                    <li class="nav-item"><a class="nav-link js-scroll-trigger" href="../projects.html">Projects</a></li>
                    <li class="nav-item"><a class="nav-link js-scroll-trigger" href="../resume.html">Resume</a></li>
                    <li class="nav-item"><a class="nav-link js-scroll-trigger" href="../contact.html">Contact</a></li>
                    <li class="nav-item"><a class="nav-link js-scroll-trigger" href="../blog.html">Blog</a></li>
                </ul>
            </div>
        </div>
    </nav>
`)
}
    