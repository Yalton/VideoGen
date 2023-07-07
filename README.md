# VideoGen 
<a name="readme-top"></a>

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

<br />
<div align="center">
  <a href="https://github.com/Yalton/VideoGen">
    <img src="doc/images/pepe.png" alt="Logo" width="80" height="80">
  </a>
  <h3 align="center">VideoGen - AI Video Generation tool</h3>
  <p align="center">
    A versatile voice assistant designed to run on a Raspberry Pi, offering a wide range of features and integrations.
    <br />
    <a href="https://github.com/Yalton/VideoGen"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/Yalton/VideoGen">View Demo</a>
    ·
    <a href="https://github.com/Yalton/VideoGen/issues">Report Bug</a>
    ·
    <a href="https://github.com/Yalton/VideoGen/issues">Request Feature</a>
  </p>
</div>
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

## About The Project

VideoGen is as protype AI Video generation tool; largely used as an experiment to learn langchain and determine how easy it would be to generate AI content en masse 

> Verdict thus far; it's rather easy 

VideoGen is in it's early alpha stages right now, and is prone to bugs; the agent will get confused on occasion depsite all of the guard rails which have been set for it. It closely resembles a child repeatadly falling from the play structure onto the sand below despite there being rails to guide it to the slide. 

When it does get to the slide so to speak; it is quite something to behold. 

<p align="right">(<a href="#readme-top">back to top</a>)</p>


### Built With

* [![Python][python-badge]][Python-url]


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

Clone the repository to your local system 

### Prerequisites

You will require some local system packages to make the script function properly 
* apt
  ```sh
  apt install ffmpeg
  ```

### Installation

1. Get a free API Key at [https://example.com](https://example.com)
2. Clone the repo
   ```sh
   git clone https://github.com/Yalton/VideoGen.git
   ```
3. Install NPM packages
   ```sh
   pip install -r requirements.txt
   ```
3. Rename tempalte.config.ini to config.ini
4. Enter your API keys in config.ini
   ```
        [API_KEYS]
        OpenAI = API_KEY
   ```
5. Setup the [stable diffusion webUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) either on the system running this, or on another local system 
   1. Specify the --api command line argument 
   2. If ran on another local system specify the URL of the Stable diffusion webui, something like 192.168.x.x:7860

6. Enter your Stable diffusion WebUI in config.ini
   ```
    [ENV_VARS]
    StableDiffusionWebUI = WEBUI_IP
   ```
<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

Simply Run videoGen.py via 

```
python3 videoGen.py
```

And watch it work 

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ROADMAP -->
## Roadmap

- [ ] Integrate controlnet to get graphics that are more than just still images
- [ ] Add subtitles to the final video via burning in a generated .srt file 
- [ ] Improve overall generation (Using GPT4 might solve this)



See the [open issues](https://github.com/Yalton/VideoGen/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->
## Contact

Dalton Bailey - [@yalt7117](https://twitter.com/yalt7117) - drbailey117@gmail.com

Project Link: [https://github.com/Yalton/VideoGen](https://github.com/Yalton/VideoGen)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* [HomeAssistantDocs](https://developers.home-assistant.io/docs/api/rest/)
<!-- * []()
* []() -->

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[python-badge]: https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white
[python-url]: https://www.python.org
[contributors-shield]: https://img.shields.io/github/contributors/Yalton/VideoGen.svg?style=for-the-badge
[contributors-url]: https://github.com/Yalton/VideoGen/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/Yalton/VideoGen.svg?style=for-the-badge
[forks-url]: https://github.com/Yalton/VideoGen/network/members
[stars-shield]: https://img.shields.io/github/stars/Yalton/VideoGen.svg?style=for-the-badge
[stars-url]: https://github.com/Yalton/VideoGen/stargazers
[issues-shield]: https://img.shields.io/github/issues/Yalton/VideoGen.svg?style=for-the-badge
[issues-url]: https://github.com/Yalton/VideoGen/issues
[license-shield]: https://img.shields.io/github/license/Yalton/VideoGen.svg?style=for-the-badge
[license-url]: https://github.com/Yalton/VideoGen/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/dalton-r-bailey
[product-screenshot]: images/screenshot.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 