# Compiling the PYNQ image
For this project we modified the PYNQ image to boot with 1024MB of CMA instead of 512MB. Xilinx recomends at CMA of 512MB for the Deep Processing Unit (DPU) alone. As this project involves several other IPs we found it necessary to increase the CMA size.

## Getting Started
Very little modification is needed from the steps provided [here](https://pynq.readthedocs.io/en/v3.0.0/pynq_sd_card.html). Only one file needs to be modified after cloning the [PYNQ](https://github.com/Xilinx/PYNQ) repository.

In `PYNQ/boards/ZCU104/petalinux_bsp/meta-user/recipes-bsp/device-tree/files/system-user.dtsi` change the amount of reserved-memory from `0x20000000` to `0x40000000`. This will increase the CMA size from 512MB to 1024MB.

**Before**
```
reserved-memory {
    #address-cells = <2>;
    #size-cells = <2>;
    ranges;
    linux,cma {
        compatible = "shared-dma-pool";
        reusable;
        size=<0x0 0x20000000>;
        alignment = <0x0 0x2000>;
        linux,cma-default;
    };
};
```

**After**
```
reserved-memory {
    #address-cells = <2>;
    #size-cells = <2>;
    ranges;
    linux,cma {
        compatible = "shared-dma-pool";
        reusable;
        size=<0x0 0x40000000>;
        alignment = <0x0 0x2000>;
        linux,cma-default;
    };
};
```

In order to aid the process of compiling PYNQ v3.0.1 we provide a docker image with documentation [here](https://hub.docker.com/r/jjsuperpower/pynq-sdbuild-env) that simplifies the host setup process.