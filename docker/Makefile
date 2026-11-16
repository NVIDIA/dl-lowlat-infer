# Extract from Dockerfile with better error handling
NGC_IMAGE ?= $(shell grep '^ARG NGC_IMAGE=' Dockerfile | sed -n 's/^ARG NGC_IMAGE=\(.*\)/\1/p' | tr -d '"')
NGC_TAG   ?= $(shell grep '^ARG NGC_TAG=' Dockerfile | sed -n 's/^ARG NGC_TAG=\(.*\)/\1/p' | tr -d '"')

# common docker variables
DOCKER_IMAGE       ?= dl-lowlat-infer
DOCKER_TAG         ?= $(if $(STAGE),$(STAGE)-)$(NGC_TAG)
DOCKER_IMAGE_TAG   ?= $(DOCKER_IMAGE):$(DOCKER_TAG)
APP_DIR            ?= $(shell grep '^ARG APP_DIR=' Dockerfile | grep -o '=.*' | tr -d '="')
SRC_DIR            ?= $(shell grep '^ARG SRC_DIR=' Dockerfile | grep -o '=.*' | tr -d '="')
GDRCOPY_DIR        ?= $(shell grep '^ARG GDRCOPY_DIR=' Dockerfile | grep -o '=.*' | tr -d '="')

# build options
DOCKER_BUILD_OPTS  ?= --pull --load
DOCKER_BUILD_ARGS  ?=
DOCKER_PROGRESS    ?= auto
CMAKE_BUILD_TYPE   ?=
CPU_ARCH           ?=
CUDA_ARCHS         ?=
BUILD_TESTS        ?=
USE_GDRCOPY        ?=

# gdrcopy variables
MAKEFILE_DIR   ?= $(dir $(lastword $(MAKEFILE_LIST)))
KERNEL_VERSION ?= $(shell uname -r)

define check_stage
	@if [ -z "$(STAGE)" ]; then echo "Error: STAGE is not set for target: $(1)" && exit 1; fi
endef

%_build:
	$(call check_stage,$*_build)
	@echo "Building docker image: $(DOCKER_IMAGE_TAG)"
	docker buildx build $(DOCKER_BUILD_OPTS) $(DOCKER_BUILD_ARGS) \
		--progress $(DOCKER_PROGRESS) \
		$(if $(NGC_IMAGE), --build-arg BASE_IMAGE=$(NGC_IMAGE)) \
		$(if $(NGC_TAG), --build-arg BASE_TAG=$(NGC_TAG)) \
		$(if $(STAGE), --target $(STAGE)) \
		$(if $(CMAKE_BUILD_TYPE), --build-arg CMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE)) \
		$(if $(CPU_ARCH), --build-arg CPU_ARCH=$(CPU_ARCH)) \
		$(if $(CUDA_ARCHS), --build-arg CUDA_ARCHS="$(CUDA_ARCHS)") \
		$(if $(BUILD_TESTS), --build-arg BUILD_TESTS=$(BUILD_TESTS)) \
		$(if $(USE_GDRCOPY), --build-arg USE_GDRCOPY=$(USE_GDRCOPY)) \
		$(if $(APP_DIR), --build-arg APP_DIR=$(APP_DIR)) \
		$(if $(SRC_DIR), --build-arg SRC_DIR=$(SRC_DIR)) \
		$(if $(GDRCOPY_DIR), --build-arg GDRCOPY_DIR=$(GDRCOPY_DIR)) \
		$(if $(KERNEL_VERSION), --build-arg KERNEL_VERSION=$(KERNEL_VERSION)) \
		--file Dockerfile \
		--tag $(DOCKER_IMAGE_TAG) \
		..

# local user options
LOCAL_USER ?= 0
ifeq ($(LOCAL_USER),1)
TAG_SUFFIX ?= -$(USER_NAME)
endif
USER_ID    ?= $(shell id --user)
USER_NAME  ?= $(shell id --user --name)
GROUP_ID   ?= $(shell id --group)
GROUP_NAME ?= $(shell id --group --name)

define add_local_user
	docker build \
		--progress $(DOCKER_PROGRESS) \
		--build-arg BASE_IMAGE_WITH_TAG=$(1) \
		--build-arg USER_ID=$(USER_ID) \
		--build-arg USER_NAME=$(USER_NAME) \
		--build-arg GROUP_ID=$(GROUP_ID) \
		--build-arg GROUP_NAME=$(GROUP_NAME) \
		--file user.dockerfile \
		--tag $(1)$(TAG_SUFFIX) \
		..
endef

# run options
DOCKER_RUN_OPTS ?= --rm -it --ipc=host
DOCKER_RUN_ARGS ?=

# Check if NVIDIA_VISIBLE_DEVICES is set and not empty
NVIDIA_VISIBLE_DEVICES_VAL = $(shell echo $$NVIDIA_VISIBLE_DEVICES)
ifeq ($(NVIDIA_VISIBLE_DEVICES_VAL),)
  # If empty or not set, use all GPUs
  GPU_OPTS      ?= --gpus=all
else
  # If set, use the specified devices
  GPU_OPTS      ?= --gpus='"device=$(NVIDIA_VISIBLE_DEVICES_VAL)"'
endif

SOURCE_DIR       ?= $(shell readlink -f ..)
CODE_DIR         ?= /code/dl-lowlat-infer
CODE_VOLUME      ?= --volume $(SOURCE_DIR):$(CODE_DIR)
CACHE_VOLUME     ?= $(if $(and $(filter 1,$(LOCAL_USER)),$(shell [ -w "$(USER_CACHE_DIR)" ] && echo 1)),--volume $(USER_CACHE_DIR):/home/$(USER_NAME)/.cache:rw)
EXTRA_VOLUMES    ?=
USER_CACHE_DIR   ?= $(shell readlink -f "${HOME}/.cache")
RUN_CMD          ?=
CONTAINER_NAME   ?= dl-lowlat-infer
WORK_DIR         ?= $(CODE_DIR)
BUILD_BEFORE_RUN ?= 1

ifeq ($(BUILD_BEFORE_RUN),1)
%_run: %_build
else
%_run:
endif
	$(call check_stage,$*_run)
ifeq ($(LOCAL_USER),1)
	$(call add_local_user,$(DOCKER_IMAGE_TAG))
endif
	@echo "Running docker image: $(DOCKER_IMAGE_TAG)"
	docker run $(DOCKER_RUN_OPTS) $(DOCKER_RUN_ARGS) $(GPU_OPTS) \
    		$(CODE_VOLUME) \
    		$(CACHE_VOLUME) \
    		$(EXTRA_VOLUMES) \
    		--workdir $(WORK_DIR) \
    		--hostname $(shell hostname)-$* \
    		--name $(CONTAINER_NAME)-$*-$(USER_NAME) \
    		--tmpfs /tmp:exec \
    		$(DOCKER_IMAGE_TAG)$(TAG_SUFFIX) $(RUN_CMD)

develop_%: STAGE = develop

cmake_%: STAGE = cmake
cmake_run: WORK_DIR = $(SRC_DIR)

release_%: STAGE = release
release_run: WORK_DIR = $(APP_DIR)

test_%: STAGE = cmake
test_%: BUILD_TESTS = ON
test_%: DOCKER_TAG = test-$(NGC_TAG)
test_run: RUN_CMD = bash -c "cd $(SRC_DIR)/build && ctest --output-on-failure"
test_run: CODE_VOLUME =

models_%: STAGE = models

build: develop_build ;

run: develop_run ;

.PHONY: build push run all clean test

all: build

clean:
	docker images -q $(DOCKER_IMAGE) | xargs -r docker rmi -f

test: test_run ;
