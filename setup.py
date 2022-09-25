if __name__ == "__main__":
    from numpy.distutils.core import setup
    from numpy.distutils.misc_util import Configuration
    
    #---------------------------------------------------------------
    """
    Building and install x3d
    """

    config = Configuration(package_name='x3d_post',
                            description="Package containing post-processing routines for my version of x3d",
                            package_path="x3d_post")
    
    config.add_subpackage(subpackage_name='post',
                        subpackage_path="x3d_post/post")  

    setup(**config.todict())
