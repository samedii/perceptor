def load_image2(img_path, img_height=None, img_width=None):

    image = Image.open(img_path)
    if img_width is not None:
        image = image.resize(
            (img_width, img_height)
        )  # change image size to (3, img_size, img_size)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    image = transform(image)[:3, :, :].unsqueeze(0)

    return image
