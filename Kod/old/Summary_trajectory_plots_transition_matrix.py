import pandas as pd
from matplotlib import pyplot as plt
import os
from Histograms import last, load_task_names_levels


def trajectory_through_tasks(data_path, tasks_path, save_pictures_path, task_id, time_limit, order_limit, cut_high_times, loglog, show, save, output):
    data = pd.read_csv(data_path)
    task_names_levels = load_task_names_levels(tasks_path=tasks_path)
    if cut_high_times:
        insert = ", time delta cut to 30 sec."
        data.update(pd.DataFrame({"time_delta_from_prev": [i if i <= 30 else 30 for i in data["time_delta_from_prev"]]}))
    else:
        insert = ""
    #print(data)
    data = data.groupby("task_session").agg({"task": "max",
                                             "student": "max",
                                             "correct": last,
                                             "order": "count",
                                             "time_delta_from_prev": "sum"})

    filtered_data = data[data.task == task_id]
    filtered_data = filtered_data.drop_duplicates(subset="student")
    time_median = filtered_data.time_delta_from_prev.median()
    order_median = filtered_data.order.median()

    filtered_data_qm = filtered_data[(filtered_data.time_delta_from_prev < time_median) & (filtered_data.order >= order_median)]
    filtered_data_qf = filtered_data[(filtered_data.time_delta_from_prev < time_median) & (filtered_data.order < order_median)]
    filtered_data_sm = filtered_data[(filtered_data.time_delta_from_prev >= time_median) & (filtered_data.order >= order_median)]
    filtered_data_sf = filtered_data[(filtered_data.time_delta_from_prev >= time_median) & (filtered_data.order < order_median)]

    print("According to task:", task_id, task_names_levels[task_id]["level"], task_names_levels[task_id]["name"])
    output.write("According to task: {} {} {}\n".format(task_id, task_names_levels[task_id]["level"], task_names_levels[task_id]["name"]))
    """
    print("total sessions", len(filtered_data))
    print("time median:", time_median, "order median", order_median, "\n")
    print("many", len(filtered_data[filtered_data.order >= order_median]))
    print("few", len(filtered_data[filtered_data.order < order_median]))
    print("quick", len(filtered_data[filtered_data.time_delta_from_prev < time_median]))
    print("slow", len(filtered_data[filtered_data.time_delta_from_prev >= time_median]), "\n")
    print("quick many", len(filtered_data_qm))
    print("quick few", len(filtered_data_qf))
    print("slow many", len(filtered_data_sm))
    print("slow few", len(filtered_data_sf), "\n")
    """

    matrix = [0 for _ in range(16)]

    for task in sorted(data.task.unique()):
        # print(task)
        task_data = data[(data.task == task)]
        # TODO: POZOR, NEJSOU OREZANE!
        task_data_cut = task_data #[(task_data.time_delta_from_prev < time_limit) & (task_data.order < order_limit)]

        time_median = task_data_cut.time_delta_from_prev.median()
        order_median = task_data_cut.order.median()

        task_data_cut_qf_cor = task_data_cut[(task_data_cut.student.isin(list(filtered_data_qf.student))) & (task_data_cut.correct == True)]  # greens
        task_data_cut_qf_inc = task_data_cut[(task_data_cut.student.isin(list(filtered_data_qf.student))) & (task_data_cut.correct != True)]
        task_data_cut_qm_cor = task_data_cut[(task_data_cut.student.isin(list(filtered_data_qm.student))) & (task_data_cut.correct == True)]  # blues
        task_data_cut_qm_inc = task_data_cut[(task_data_cut.student.isin(list(filtered_data_qm.student))) & (task_data_cut.correct != True)]
        task_data_cut_sf_cor = task_data_cut[(task_data_cut.student.isin(list(filtered_data_sf.student))) & (task_data_cut.correct == True)]  # yellows
        task_data_cut_sf_inc = task_data_cut[(task_data_cut.student.isin(list(filtered_data_sf.student))) & (task_data_cut.correct != True)]
        task_data_cut_sm_cor = task_data_cut[(task_data_cut.student.isin(list(filtered_data_sm.student))) & (task_data_cut.correct == True)]  # reds
        task_data_cut_sm_inc = task_data_cut[(task_data_cut.student.isin(list(filtered_data_sm.student))) & (task_data_cut.correct != True)]

        task_data_cut_no_cor = task_data_cut[(~task_data_cut.student.isin(list(filtered_data_qf.student))) &
                                             (~task_data_cut.student.isin(list(filtered_data_qm.student))) &
                                             (~task_data_cut.student.isin(list(filtered_data_sf.student))) &
                                             (~task_data_cut.student.isin(list(filtered_data_sm.student))) &
                                             (task_data_cut.correct == True)]

        task_data_cut_no_inc = task_data_cut[(~task_data_cut.student.isin(list(filtered_data_qf.student))) &
                                             (~task_data_cut.student.isin(list(filtered_data_qm.student))) &
                                             (~task_data_cut.student.isin(list(filtered_data_sf.student))) &
                                             (~task_data_cut.student.isin(list(filtered_data_sm.student))) &
                                             (task_data_cut.correct != True)]

        matrix[0] += len(task_data_cut_qf_cor[(task_data_cut_qf_cor.time_delta_from_prev < time_median) &
                                              (task_data_cut_qf_cor.order < order_median)]) + \
                     len(task_data_cut_qf_inc[(task_data_cut_qf_inc.time_delta_from_prev < time_median) &
                                              (task_data_cut_qf_inc.order < order_median)])
        matrix[1] += len(task_data_cut_qf_cor[(task_data_cut_qf_cor.time_delta_from_prev < time_median) &
                                              (task_data_cut_qf_cor.order >= order_median)]) + \
                     len(task_data_cut_qf_inc[(task_data_cut_qf_inc.time_delta_from_prev < time_median) &
                                              (task_data_cut_qf_inc.order >= order_median)])
        matrix[2] += len(task_data_cut_qf_cor[(task_data_cut_qf_cor.time_delta_from_prev >= time_median) &
                                              (task_data_cut_qf_cor.order < order_median)]) + \
                     len(task_data_cut_qf_inc[(task_data_cut_qf_inc.time_delta_from_prev >= time_median) &
                                              (task_data_cut_qf_inc.order < order_median)])
        matrix[3] += len(task_data_cut_qf_cor[(task_data_cut_qf_cor.time_delta_from_prev >= time_median) &
                                              (task_data_cut_qf_cor.order >= order_median)]) + \
                     len(task_data_cut_qf_inc[(task_data_cut_qf_inc.time_delta_from_prev >= time_median) &
                                              (task_data_cut_qf_inc.order >= order_median)])

        matrix[4] += len(task_data_cut_qm_cor[(task_data_cut_qm_cor.time_delta_from_prev < time_median) &
                                              (task_data_cut_qm_cor.order < order_median)]) + \
                     len(task_data_cut_qm_inc[(task_data_cut_qm_inc.time_delta_from_prev < time_median) &
                                              (task_data_cut_qm_inc.order < order_median)])
        matrix[5] += len(task_data_cut_qm_cor[(task_data_cut_qm_cor.time_delta_from_prev < time_median) &
                                              (task_data_cut_qm_cor.order >= order_median)]) + \
                     len(task_data_cut_qm_inc[(task_data_cut_qm_inc.time_delta_from_prev < time_median) &
                                              (task_data_cut_qm_inc.order >= order_median)])
        matrix[6] += len(task_data_cut_qm_cor[(task_data_cut_qm_cor.time_delta_from_prev >= time_median) &
                                              (task_data_cut_qm_cor.order < order_median)]) + \
                     len(task_data_cut_qm_inc[(task_data_cut_qm_inc.time_delta_from_prev >= time_median) &
                                              (task_data_cut_qm_inc.order < order_median)])
        matrix[7] += len(task_data_cut_qm_cor[(task_data_cut_qm_cor.time_delta_from_prev >= time_median) &
                                              (task_data_cut_qm_cor.order >= order_median)]) + \
                     len(task_data_cut_qm_inc[(task_data_cut_qm_inc.time_delta_from_prev >= time_median) &
                                              (task_data_cut_qm_inc.order >= order_median)])

        matrix[8] += len(task_data_cut_sf_cor[(task_data_cut_sf_cor.time_delta_from_prev < time_median) &
                                              (task_data_cut_sf_cor.order < order_median)]) + \
                     len(task_data_cut_sf_inc[(task_data_cut_sf_inc.time_delta_from_prev < time_median) &
                                              (task_data_cut_sf_inc.order < order_median)])
        matrix[9] += len(task_data_cut_sf_cor[(task_data_cut_sf_cor.time_delta_from_prev < time_median) &
                                              (task_data_cut_sf_cor.order >= order_median)]) + \
                     len(task_data_cut_sf_inc[(task_data_cut_sf_inc.time_delta_from_prev < time_median) &
                                              (task_data_cut_sf_inc.order >= order_median)])
        matrix[10] += len(task_data_cut_sf_cor[(task_data_cut_sf_cor.time_delta_from_prev >= time_median) &
                                               (task_data_cut_sf_cor.order < order_median)]) + \
                      len(task_data_cut_sf_inc[(task_data_cut_sf_inc.time_delta_from_prev >= time_median) &
                                               (task_data_cut_sf_inc.order < order_median)])
        matrix[11] += len(task_data_cut_sf_cor[(task_data_cut_sf_cor.time_delta_from_prev >= time_median) &
                                               (task_data_cut_sf_cor.order >= order_median)]) + \
                      len(task_data_cut_sf_inc[(task_data_cut_sf_inc.time_delta_from_prev >= time_median) &
                                               (task_data_cut_sf_inc.order >= order_median)])

        matrix[12] += len(task_data_cut_sm_cor[(task_data_cut_sm_cor.time_delta_from_prev < time_median) &
                                               (task_data_cut_sm_cor.order < order_median)]) + \
                      len(task_data_cut_sm_inc[(task_data_cut_sm_inc.time_delta_from_prev < time_median) &
                                               (task_data_cut_sm_inc.order < order_median)])
        matrix[13] += len(task_data_cut_sm_cor[(task_data_cut_sm_cor.time_delta_from_prev < time_median) &
                                               (task_data_cut_sm_cor.order >= order_median)]) + \
                      len(task_data_cut_sm_inc[(task_data_cut_sm_inc.time_delta_from_prev < time_median) &
                                               (task_data_cut_sm_inc.order >= order_median)])
        matrix[14] += len(task_data_cut_sm_cor[(task_data_cut_sm_cor.time_delta_from_prev >= time_median) &
                                               (task_data_cut_sm_cor.order < order_median)]) + \
                      len(task_data_cut_sm_inc[(task_data_cut_sm_inc.time_delta_from_prev >= time_median) &
                                               (task_data_cut_sm_inc.order < order_median)])
        matrix[15] += len(task_data_cut_sm_cor[(task_data_cut_sm_cor.time_delta_from_prev >= time_median) &
                                               (task_data_cut_sm_cor.order >= order_median)]) + \
                      len(task_data_cut_sm_inc[(task_data_cut_sm_inc.time_delta_from_prev >= time_median) &
                                               (task_data_cut_sm_inc.order >= order_median)])

        # graf
        if show or save:
            if len(task_data_cut_no_cor):
                ax9 = task_data_cut_no_cor.plot.scatter(x="time_delta_from_prev",
                                                        y="order",
                                                        c="black",
                                                        xlim=(0, time_limit),
                                                        ylim=(0, order_limit),
                                                        loglog=loglog,
                                                        use_index=False,
                                                        title="Time vs edits/submits{}\nlevel {}, task {}\n{} sessions, {} sessions shown".format(
                                                            insert, task_names_levels[task]["level"], task_names_levels[task]["name"],
                                                            len(task_data.index), len(task_data_cut.index)),
                                                        grid=True,
                                                        ax=plt.gca())
                ax9.set_xlabel("Total time spent")
                ax9.set_ylabel("Number of edits and submits")
            if len(task_data_cut_no_inc):
                ax10 = task_data_cut_no_inc.plot.scatter(x="time_delta_from_prev",
                                                         y="order",
                                                         c="0.7",
                                                         xlim=(0, time_limit),
                                                         ylim=(0, order_limit),
                                                         loglog=loglog,
                                                         use_index=False,
                                                         title="Time vs edits/submits{}\nlevel {}, task {}\n{} sessions, {} sessions shown".format(
                                                             insert, task_names_levels[task]["level"], task_names_levels[task]["name"],
                                                             len(task_data.index), len(task_data_cut.index)),
                                                         grid=True,
                                                         ax=plt.gca())
                ax10.set_xlabel("Total time spent")
                ax10.set_ylabel("Number of edits and submits")
            if len(task_data_cut_qf_cor):
                ax1 = task_data_cut_qf_cor.plot.scatter(x="time_delta_from_prev",
                                                        y="order",
                                                        c="xkcd:green",
                                                        xlim=(0, time_limit),
                                                        ylim=(0, order_limit),
                                                        loglog=loglog,
                                                        use_index=False,
                                                        title="Time vs edits/submits{}\nlevel {}, task {}\n{} sessions, {} sessions shown".format(
                                                            insert, task_names_levels[task]["level"], task_names_levels[task]["name"],
                                                            len(task_data.index), len(task_data_cut.index)),
                                                        grid=True,
                                                        ax=plt.gca())
                ax1.set_xlabel("Total time spent")
                ax1.set_ylabel("Number of edits and submits")
            if len(task_data_cut_qf_inc):
                ax2 = task_data_cut_qf_inc.plot.scatter(x="time_delta_from_prev",
                                                        y="order",
                                                        c="xkcd:light green",
                                                        xlim=(0, time_limit),
                                                        ylim=(0, order_limit),
                                                        loglog=loglog,
                                                        use_index=False,
                                                        title="Time vs edits/submits{}\nlevel {}, task {}\n{} sessions, {} sessions shown".format(
                                                            insert, task_names_levels[task]["level"], task_names_levels[task]["name"],
                                                            len(task_data.index), len(task_data_cut.index)),
                                                        grid=True,
                                                        ax=plt.gca())
                ax2.set_xlabel("Total time spent")
                ax2.set_ylabel("Number of edits and submits")
            if len(task_data_cut_qm_cor):
                ax3 = task_data_cut_qm_cor.plot.scatter(x="time_delta_from_prev",
                                                        y="order",
                                                        c="xkcd:blue",
                                                        xlim=(0, time_limit),
                                                        ylim=(0, order_limit),
                                                        loglog=loglog,
                                                        use_index=False,
                                                        title="Time vs edits/submits{}\nlevel {}, task {}\n{} sessions, {} sessions shown".format(
                                                            insert, task_names_levels[task]["level"], task_names_levels[task]["name"],
                                                            len(task_data.index), len(task_data_cut.index)),
                                                        grid=True,
                                                        ax=plt.gca())
                ax3.set_xlabel("Total time spent")
                ax3.set_ylabel("Number of edits and submits")
            if len(task_data_cut_qm_inc):
                ax4 = task_data_cut_qm_inc.plot.scatter(x="time_delta_from_prev",
                                                        y="order",
                                                        c="xkcd:light blue",
                                                        xlim=(0, time_limit),
                                                        ylim=(0, order_limit),
                                                        loglog=loglog,
                                                        use_index=False,
                                                        title="Time vs edits/submits{}\nlevel {}, task {}\n{} sessions, {} sessions shown".format(
                                                            insert, task_names_levels[task]["level"], task_names_levels[task]["name"],
                                                            len(task_data.index), len(task_data_cut.index)),
                                                        grid=True,
                                                        ax=plt.gca())
                ax4.set_xlabel("Total time spent")
                ax4.set_ylabel("Number of edits and submits")
            if len(task_data_cut_sf_cor):
                ax5 = task_data_cut_sf_cor.plot.scatter(x="time_delta_from_prev",
                                                        y="order",
                                                        c="xkcd:yellow",
                                                        xlim=(0, time_limit),
                                                        ylim=(0, order_limit),
                                                        loglog=loglog,
                                                        use_index=False,
                                                        title="Time vs edits/submits{}\nlevel {}, task {}\n{} sessions, {} sessions shown".format(
                                                            insert, task_names_levels[task]["level"], task_names_levels[task]["name"],
                                                            len(task_data.index), len(task_data_cut.index)),
                                                        grid=True,
                                                        ax=plt.gca())
                ax5.set_xlabel("Total time spent")
                ax5.set_ylabel("Number of edits and submits")
            if len(task_data_cut_sf_inc):
                ax6 = task_data_cut_sf_inc.plot.scatter(x="time_delta_from_prev",
                                                        y="order",
                                                        c="xkcd:light yellow",
                                                        xlim=(0, time_limit),
                                                        ylim=(0, order_limit),
                                                        loglog=loglog,
                                                        use_index=False,
                                                        title="Time vs edits/submits{}\nlevel {}, task {}\n{} sessions, {} sessions shown".format(
                                                            insert, task_names_levels[task]["level"], task_names_levels[task]["name"],
                                                            len(task_data.index), len(task_data_cut.index)),
                                                        grid=True,
                                                        ax=plt.gca())
                ax6.set_xlabel("Total time spent")
                ax6.set_ylabel("Number of edits and submits")
            if len(task_data_cut_sm_cor):
                ax7 = task_data_cut_sm_cor.plot.scatter(x="time_delta_from_prev",
                                                        y="order",
                                                        c="xkcd:red",
                                                        xlim=(0, time_limit),
                                                        ylim=(0, order_limit),
                                                        loglog=loglog,
                                                        use_index=False,
                                                        title="Time vs edits/submits{}\nlevel {}, task {}\n{} sessions, {} sessions shown".format(
                                                            insert, task_names_levels[task]["level"], task_names_levels[task]["name"],
                                                            len(task_data.index), len(task_data_cut.index)),
                                                        grid=True,
                                                        ax=plt.gca())
                ax7.set_xlabel("Total time spent")
                ax7.set_ylabel("Number of edits and submits")
            if len(task_data_cut_sm_inc):
                ax8 = task_data_cut_sm_inc.plot.scatter(x="time_delta_from_prev",
                                                        y="order",
                                                        c="xkcd:light red",
                                                        xlim=(0, time_limit),
                                                        ylim=(0, order_limit),
                                                        loglog=loglog,
                                                        use_index=False,
                                                        title="Time vs edits/submits{}\nlevel {}, task {}\n{} sessions, {} sessions shown".format(
                                                            insert, task_names_levels[task]["level"], task_names_levels[task]["name"],
                                                            len(task_data.index), len(task_data_cut.index)),
                                                        grid=True,
                                                        ax=plt.gca())
                ax8.set_xlabel("Total time spent")
                ax8.set_ylabel("Number of edits and submits")
            if show:
                plt.show()
            if save:
                if not os.path.exists("{} {}".format(save_pictures_path, task_id)):
                    os.makedirs("{} {}".format(save_pictures_path, task_id))
                plt.savefig("{} {}/time_vs_edits_trajectory {} {}.png".format(save_pictures_path, task_id,
                                                                              task_names_levels[task]["level"],
                                                                              task_names_levels[task]["name"]))
            plt.clf()

        """
        print(len(task_data_cut))
        sfc = len(task_data_cut_sf_cor)
        sfi = len(task_data_cut_sf_inc)
        smc = len(task_data_cut_sm_cor)
        smi = len(task_data_cut_sm_inc)
        qfc = len(task_data_cut_qf_cor)
        qfi = len(task_data_cut_qf_inc)
        qmc = len(task_data_cut_qm_cor)
        qmi = len(task_data_cut_qm_inc)
        noc = len(task_data_cut_no_cor)
        noi = len(task_data_cut_no_inc)
        print(sfc, sfi, smc, smi, qfc, qfi, qmc, qmi)
        print(noc, noi)
        print(sfc + sfi + smc + smi + qfc + qfi + qmc + qmi)
        print(noc + noi)
        print(sfc + sfi + smc + smi + qfc + qfi + qmc + qmi + noc + noi)
        """

    # relativizace matice
    rel_matrix = [0 for _ in range(16)]
    sums = [0 for _ in range(4)]
    zero_sums = [0 for _ in range(4)]
    for i in range(4):
        sums[i] = sum(matrix[4*i:4*i+4])
        if not sums[i]:
            zero_sums[i] += 1

    for i in range(16):
        if sums[i//4]:
            rel_matrix[i] = matrix[i] / sums[i//4]
        else:
            rel_matrix[i] = None

    names = ("qf", "qm", "sf", "sm")
    output.write("Absolute matrix:\n")
    for i in range(4):
        output.write("{} {}\n".format(names[i], matrix[4 * i:4 * i + 4]))
    output.write("Relative matrix:\n")
    for i in range(4):
        output.write("{} {}\n".format(names[i], rel_matrix[4 * i:4 * i + 4]))
    output.write("\n")
    return matrix, rel_matrix, zero_sums


def overall_trajectory(output_path, num_of_tasks):
    with open(output_path, "w") as output:
        output.write("Format of matrix: [[quick_few_operations, quick_many_operations], [slow_few_operations, slow_many_operations]]\n")
        total_matrix = [0 for _ in range(16)]
        avg_rel_matrix = [0 for _ in range(16)]
        total_zero_sums = [0 for _ in range(4)]
        for task_id in range(1, num_of_tasks+1):
            matrix, rel_matrix, zero_sums = trajectory_through_tasks(data_path="C:/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-02-10/joined_snapshots_tasks_delta.csv",
                                                          tasks_path="C:/Dokumenty/Matej/MUNI/Diplomka/Data/robomission-2018-02-10/tasks.csv",
                                                          save_pictures_path="C:/Dokumenty/Matej/MUNI/Diplomka/Obrazky/Cas vs edity trajektorie/Cas vs edity trajektorie",
                                                          task_id=task_id,
                                                          time_limit=600,
                                                          order_limit=100,
                                                          cut_high_times=True,
                                                          loglog=True,
                                                          show=False,
                                                          save=False,
                                                          output=output)
            for i in range(16):
                total_matrix[i] += matrix[i]
                if rel_matrix[i]:
                    avg_rel_matrix[i] += rel_matrix[i]
            for i in range(4):
                total_zero_sums[i] += zero_sums[i]

        relativised_total_matrix = [0 for _ in range(16)]
        total_sums = [0 for _ in range(4)]
        for i in range(4):
            total_sums[i] = sum(total_matrix[4*i:4*i+4])


        for i in range(16):
            if total_sums[i // 4]:
                relativised_total_matrix[i] = total_matrix[i] / total_sums[i // 4]
            else:
                relativised_total_matrix[i] = None
            avg_rel_matrix[i] /= num_of_tasks - total_zero_sums[i // 4]

        names = ("qf", "qm", "sf", "sm")
        output.write("\nTotal absolute matrix:\n")
        for i in range(4):
            output.write("{} {}\n".format(names[i], total_matrix[4 * i:4 * i + 4]))
        output.write("\nRelative total absolute matrix:\n")
        for i in range(4):
            output.write("{} {}\n".format(names[i], relativised_total_matrix[4*i:4*i+4]))
        output.write("\nAverage relative matrix:\n")
        for i in range(4):
            output.write("{} {}\n".format(names[i], avg_rel_matrix[4 * i:4 * i + 4]))


overall_trajectory(output_path="C:/Dokumenty/Matej/MUNI/Diplomka/Obrazky/Cas vs edity trajektorie/trajektorie3.txt",
                   num_of_tasks=86)