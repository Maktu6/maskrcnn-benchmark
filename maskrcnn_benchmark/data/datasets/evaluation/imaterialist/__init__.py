from .imaterialist_eval import do_imaterialist_evaluation


def imaterialist_evaluation(
    dataset,
    predictions,
    output_folder,
    box_only,
    iou_types,
    expected_results,
    expected_results_sigma_tol,
    maskiou_on=False,
):
    return do_imaterialist_evaluation(
        dataset=dataset,
        predictions=predictions,
        box_only=box_only,
        output_folder=output_folder,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
        maskiou_on=maskiou_on,
    )
